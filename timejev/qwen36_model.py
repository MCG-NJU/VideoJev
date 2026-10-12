"""Independent Qwen3.6 MoE grounding model; no changes to the dense 4B path.

The replicated, frozen native ViT is deliberately outside the ZeRO engine.
Different ranks can therefore encode different numbers of RGB chunks without
issuing mismatched parameter-gather collectives. Features live only in the
current batch and are never serialized as a reusable vision cache.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from transformers import AutoConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeModel, Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel, is_fast_path_available,
)

from .decision_inputs import FORMULATION, SPECIAL_TOKENS, load_processor, token_metadata
from .decision_model import typed_outputs
from .model import PointerHead
from .qwen36_load import load_hf_submodule_weights
from .qwen36_moe_ops import install_xtuner_grouped_experts, install_fp32_routers
from .qwen36_recipe import validate_recipe
from .losses import decision_losses
from .vision import VISUAL_PROTOCOL


class AttentionOnlyMoeDecoderLayer(Qwen3_5MoeDecoderLayer):
    """Identical final attention + residual readout to the 4B baseline."""

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        return residual + hidden_states


class NativeMRope:
    """Use official Qwen multimodal position methods without constructing an LLM."""
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids
    get_rope_index = Qwen3_5MoeModel.get_rope_index

    def __init__(self, config):
        self.config = config


class Qwen36Grounder(nn.Module):
    visual_protocol = VISUAL_PROTOCOL
    formulation = FORMULATION

    def __init__(self, language, native_config, config, tokenizer):
        super().__init__()
        self.language_model = language
        self.native_config, self.recipe = native_config, dict(config)
        self.positions = NativeMRope(native_config)
        self.freeze_llm = bool(config['freeze_llm'])
        self.video_token_id = native_config.video_token_id
        self.head = PointerHead(native_config.text_config.hidden_size, config['head_dim'],
                                similarity='cosine', logit_scale=config['head_logit_scale'])
        self.decision_metadata = token_metadata(tokenizer) | dict(
            attention_mode='causal', grounding_candidates='timestamp_options',
            readout=dict(version='timejev_readout_v1', readout_mode='last_attention',
                         original_decoder_layers=native_config.text_config.num_hidden_layers,
                         executed_decoder_layers=native_config.text_config.num_hidden_layers,
                         final_language_norm=True, final_mlp=False, head='original_pointer'),
            head_scoring=dict(similarity='cosine', logit_scale=10.0, trainable_scale=False,
                              normalization='L2 after Q/K projection', eps=1e-6, trainable_bias=True),
            router_compute_dtype=config['router_compute_dtype'],
            model_id=config.get('model_id'), model_revision=config.get('model_revision'))
        if self.freeze_llm:
            language.requires_grad_(False)
            language.gradient_checkpointing_disable()
            language.eval()
        elif config.get('checkpointing', True):
            language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_llm:
            self.language_model.eval()
        return self


    def _hidden(self, input_ids, types, grids, lengths, visual_features):
        if not lengths:
            # All ranks execute the same layer/gather schedule on partial global
            # batches. The head makes this placeholder's loss exactly zero.
            input_ids = input_ids[:, :64]
            lengths = [input_ids.shape[1]]
            position_ids = torch.arange(lengths[0], device=input_ids.device)[None, None].expand(3, 1, -1)
        else:
            input_ids, types = input_ids[:, :sum(lengths)], types[:, :sum(lengths)]
            position_pieces, offset = [], 0
            for index, length in enumerate(lengths):
                positions, _ = self.positions.get_rope_index(
                    input_ids[:, offset:offset + length], types[:, offset:offset + length],
                    video_grid_thw=grids[index:index + 1])
                position_pieces.append(positions)
                offset += length
            position_ids = torch.cat(position_pieces, dim=-1)
        with torch.set_grad_enabled(torch.is_grad_enabled()
                                    and not self.freeze_llm):
            embeds = self.language_model.embed_tokens(input_ids)
            mask = input_ids == self.video_token_id
            if int(mask.sum()) != len(visual_features):
                raise ValueError('Every native video token requires exactly one current RGB embedding')
            if len(visual_features):
                embeds = embeds.masked_scatter(mask[..., None].expand_as(embeds), visual_features.to(embeds))
            sizes = torch.tensor(lengths, dtype=torch.int32, device=embeds.device)
            cu = torch.cat((sizes.new_zeros(1), sizes.cumsum(0, dtype=torch.int32)))
            seq_idx = torch.repeat_interleave(torch.arange(len(lengths), device=embeds.device,
                                                           dtype=torch.int32), sizes.long())[None]
            # No dense attention mask; FA varlen and FLA/conv boundaries isolate
            # complete episodes. Training turns within an episode stay causal.
            output = self.language_model(
                inputs_embeds=embeds, position_ids=position_ids,
                attention_mask={'full_attention': None, 'linear_attention': None},
                use_cache=False, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                max_length_q=max(lengths), max_length_k=max(lengths), seq_idx=seq_idx)
            return output.last_hidden_state[0]

    def forward(self, input_ids, mm_token_type_ids, video_grid_thw, pair_indices,
                query_indices, score_mask, decision_tasks, rgb_visual_features,
                labels=None, score_durations=None, episode_lengths=None,
                packed_sample_lengths=None, **unused):
        if input_ids.shape[0] != 1:
            raise ValueError('Qwen3.6 expects one unpadded packed token stream per microbatch')
        lengths = episode_lengths if episode_lengths is not None else packed_sample_lengths
        if lengths is None or len(lengths) != len(video_grid_thw):
            raise ValueError('Each complete episode/question must own one native video grid')
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if input_ids.is_cuda else contextlib.nullcontext()
        with amp:
            hidden = self._hidden(input_ids, mm_token_type_ids, video_grid_thw, lengths, rgb_visual_features)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            if not lengths:
                zero = self.head(hidden[62][None, None], hidden[63][None]).sum() * 0.0
                logits = zero.expand(score_mask.shape)
            else:
                logits = self.head(hidden[pair_indices], hidden[query_indices])
            return decision_losses(logits, score_mask, decision_tasks, labels,
                score_durations=score_durations,
                soft_iou_weight=self.recipe.get('soft_iou_weight', 0.))



def build_grounder(config, device, *, ds_config=None, load_pretrained=True, native_config=None):
    """Create only sharded text weights; frozen vision is built independently."""
    native = native_config or AutoConfig.from_pretrained(config['base'], local_files_only=True)
    if native.model_type != 'qwen3_5_moe':
        raise ValueError('Expected the official Qwen3.6 Qwen3_5Moe checkpoint')
    native.text_config._attn_implementation = 'flash_attention_2'
    native.text_config.use_cache = False
    if not is_fast_path_available:
        raise RuntimeError('Native FLA and causal-conv1d are required; slow packing fallback is forbidden')
    zero3 = ds_config is not None
    if zero3:
        import deepspeed
        init_context = deepspeed.zero.Init(config_dict_or_path=ds_config, dtype=torch.float32)
    else:
        init_context = contextlib.nullcontext()
    with init_context, no_init_weights():
        language = Qwen3_5MoeTextModel(native.text_config)
    final = language.layers[-1]
    if final.block_type != 'full_attention':
        raise ValueError('The baseline last-attention readout requires a final full-attention layer')
    final.__class__ = AttentionOnlyMoeDecoderLayer
    del final.mlp
    del final.post_attention_layernorm
    if load_pretrained:
        load_hf_submodule_weights(language, config['base'], prefix='model.language_model.', zero3=zero3)
    install_xtuner_grouped_experts(language)
    install_fp32_routers(language)
    tokenizer = load_processor(config['base']).tokenizer
    if len(tokenizer) > native.text_config.vocab_size:
        raise ValueError('Decision tokens exceed native padded embedding rows; explicit resize is required')
    model = Qwen36Grounder(language, native, config, tokenizer)
    if load_pretrained:
        embedding = language.embed_tokens.weight
        context = (deepspeed.zero.GatheredParameters([embedding], modifier_rank=0)
                   if zero3 else contextlib.nullcontext())
        with context, torch.no_grad():
            if not zero3 or torch.distributed.get_rank() == 0:
                initial = embedding[tokenizer.convert_tokens_to_ids('<|im_end|>')].clone()
                for token in SPECIAL_TOKENS:
                    embedding[tokenizer.convert_tokens_to_ids(token)].copy_(initial)
    model.to(device)
    return model


def build_frozen_vision(config, device):
    native = AutoConfig.from_pretrained(config['base'], local_files_only=True)
    native.vision_config._attn_implementation = 'flash_attention_2'
    with no_init_weights():
        visual = Qwen3_5MoeVisionModel(native.vision_config)
    load_hf_submodule_weights(visual, config['base'], prefix='model.visual.', zero3=False)
    visual.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    return visual


@torch.no_grad()
def encode_rgb(visual, pixel_chunks, grids, device):
    """Encode official patches in native temporal-pair chunks, with no disk cache."""
    if len(pixel_chunks) != len(grids):
        raise ValueError('RGB videos and native grids disagree')
    outputs = []
    for chunks, grid in zip(pixel_chunks, grids.tolist()):
        temporal, height, width = grid
        seen = 0
        for patches in chunks:
            if len(patches) % (height * width):
                raise ValueError('A RGB chunk cuts a native temporal pair')
            groups = len(patches) // (height * width)
            chunk_grid = torch.tensor([[groups, height, width]], device=device, dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = visual(patches.to(device=device, dtype=visual.dtype), grid_thw=chunk_grid,
                                return_dict=True).pooler_output
            outputs.append(result.detach())
            seen += groups
        if seen != temporal:
            raise ValueError('RGB chunks must preserve all original sampled temporal pairs')
    if not outputs:
        return torch.empty((0, visual.config.out_hidden_size), device=device, dtype=torch.bfloat16)
    return torch.cat(outputs)


def model_batch(batch, visual, device, *, supervised=True):
    """Keep raw pixels and reporting rows out of recursive all-at-once GPU copies."""
    features = encode_rgb(visual, batch.get('video_pixel_chunks', []), batch['video_grid_thw'], device)
    keys = ('input_ids', 'mm_token_type_ids', 'video_grid_thw', 'pair_indices', 'query_indices',
            'score_mask', 'decision_tasks', 'episode_lengths', 'packed_sample_lengths', 'score_durations')
    result = {key: (batch[key].to(device, non_blocking=True) if torch.is_tensor(batch[key]) else batch[key])
              for key in keys if key in batch}
    if supervised and 'labels' in batch:
        result['labels'] = batch['labels'].to(device, non_blocking=True)
    result['rgb_visual_features'] = features
    return result
