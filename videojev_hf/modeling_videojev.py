"""Native causal Qwen video features with the trained VideoJev decision head."""
import contextlib
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5DecoderLayer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel, Qwen3_5MoeDecoderLayer

from .configuration_videojev import VideoJevConfig
from .processing_videojev import prepare_query_groups, prepare_video, format_predictions
from .videojev_moe import install_fp32_routers, install_xtuner_grouped_experts


class AttentionOnlyDecoderLayer(Qwen3_5DecoderLayer):
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states=self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        return residual + hidden_states


class AttentionOnlyMoeDecoderLayer(Qwen3_5MoeDecoderLayer):
    forward = AttentionOnlyDecoderLayer.forward


class CosineDecisionHead(nn.Module):
    def __init__(self, hidden, dimension, scale):
        super().__init__()
        self.q = nn.Linear(hidden, dimension)
        self.k = nn.Linear(hidden, dimension)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = float(scale)

    def forward(self, pairs, queries):
        keys = F.linear(pairs.float(), self.k.weight.float(), self.k.bias.float())
        query = F.linear(queries.float(), self.q.weight.float(), self.q.bias.float())
        keys = F.normalize(keys, dim=-1, eps=1e-6)
        query = F.normalize(query, dim=-1, eps=1e-6)
        return torch.einsum('qkd,qd->qk', keys, query) * self.scale + self.bias.float()


@dataclass
class VideoJevOutput(ModelOutput):
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    score_mask: torch.Tensor | None = None


class VideoJevForVideoUnderstanding(PreTrainedModel):
    config_class = VideoJevConfig
    base_model_prefix = 'backbone'
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = ['Qwen3_5DecoderLayer', 'Qwen3_5MoeDecoderLayer',
                         'AttentionOnlyDecoderLayer', 'AttentionOnlyMoeDecoderLayer',
                         'Qwen3_5VisionBlock', 'Qwen3_5MoeVisionBlock']
    _keep_in_fp32_modules = ['head']
    _is_stateful = True

    def __init__(self, config):
        super().__init__(config)
        os.environ['FLA_INTRACARD_CP'] = '0'
        native = config.backbone_config
        native._attn_implementation = config._attn_implementation
        native.text_config._attn_implementation = config._attn_implementation
        native.vision_config._attn_implementation = config._attn_implementation
        self.is_moe = native.model_type == 'qwen3_5_moe'
        self.backbone = (Qwen3_5MoeModel if self.is_moe else Qwen3_5Model)(native)
        last = self.backbone.language_model.layers[-1]
        if last.block_type != 'full_attention':
            raise ValueError('Expected a final full-attention decoder layer')
        last.__class__ = AttentionOnlyMoeDecoderLayer if self.is_moe else AttentionOnlyDecoderLayer
        del last.mlp
        del last.post_attention_layernorm
        self.head = CosineDecisionHead(native.text_config.hidden_size, config.head_dim, config.head_logit_scale)
        if self.is_moe:
            install_fp32_routers(self.backbone.language_model)
        self.post_init()

    def prepare_for_inference(self, *, xtuner_experts=False):
        """Keep native BF16 language weights and perform sensitive scoring in FP32."""
        self.head.float()
        self.backbone.visual.float()
        if self.is_moe:
            for module in self.backbone.language_model.modules():
                if getattr(module, 'timejev_router_compute_dtype', None) == 'float32':
                    module.float()
            if xtuner_experts:
                install_xtuner_grouped_experts(self.backbone.language_model)
        return self.eval().requires_grad_(False)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @torch.inference_mode()
    def encode_video(self, pixel_values_videos, video_grid_thw, groups_per_chunk=16):
        visual = self.backbone.visual
        device = next(visual.parameters()).device
        if video_grid_thw.shape[0] != 1:
            raise ValueError('Encode one complete video at a time')
        t, h, w = video_grid_thw[0].tolist()
        if len(pixel_values_videos) != t*h*w:
            raise ValueError('Video grid and native pixel patches disagree')
        output = []
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
        with amp:
            for start in range(0, t, groups_per_chunk):
                count = min(groups_per_chunk, t-start)
                pixels = pixel_values_videos[start*h*w:(start+count)*h*w].to(device)
                grid = torch.tensor([[count, h, w]], device=device)
                output.append(visual(pixels, grid_thw=grid, return_dict=True).pooler_output)
        return torch.cat(output)

    def forward(self, input_ids, mm_token_type_ids, video_grid_thw, pair_indices,
                query_indices, score_mask, decision_tasks, visual_features=None,
                pixel_values_videos=None, attention_mask=None, **kwargs):
        if input_ids.shape[0] != 1 or video_grid_thw.shape[0] != 1:
            raise ValueError('Each call owns one complete video and its query group')
        if attention_mask is not None and not attention_mask.bool().all():
            raise ValueError('Pass an unpadded query group')
        if visual_features is None:
            if pixel_values_videos is None:
                raise ValueError('Supply native visual features or native pixel patches')
            visual_features = self.encode_video(pixel_values_videos, video_grid_thw)
        device = self.get_input_embeddings().weight.device
        input_ids, mm_token_type_ids = input_ids.to(device), mm_token_type_ids.to(device)
        grid = video_grid_thw.to(device)
        pairs, queries, mask = pair_indices.to(device), query_indices.to(device), score_mask.to(device)
        tasks = torch.as_tensor(decision_tasks, device=device)
        if tasks.shape != queries.shape or not ((tasks == 0) | (tasks == 1)).all():
            raise ValueError('decision_tasks must contain one grounding(0)/choice(1) value per question')
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
        with amp:
            embeds = self.get_input_embeddings()(input_ids)
            slots = input_ids == self.config.backbone_config.video_token_id
            if int(slots.sum()) != len(visual_features):
                raise ValueError('Every native video token needs exactly one feature')
            embeds = embeds.masked_scatter(slots[..., None].expand_as(embeds), visual_features.to(embeds))
            if self.is_moe:
                positions, _ = self.backbone.get_rope_index(input_ids, mm_token_type_ids, video_grid_thw=grid)
            else:
                positions = self.backbone.compute_3d_position_ids(input_ids=input_ids,
                    inputs_embeds=embeds, video_grid_thw=grid, attention_mask=None,
                    mm_token_type_ids=mm_token_type_ids)
            n = input_ids.shape[1]
            native = dict(inputs_embeds=embeds, position_ids=positions, use_cache=False)
            if device.type == 'cuda':
                cu = torch.tensor([0, n], device=device, dtype=torch.int32)
                native.update(attention_mask={'full_attention': None, 'linear_attention': None},
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=n, max_length_k=n,
                    seq_idx=torch.zeros((1, n), device=device, dtype=torch.int32))
            hidden = self.backbone.language_model(**native).last_hidden_state[0]
        with torch.autocast(device_type=device.type, enabled=False):
            logits = self.head(hidden[pairs], hidden[queries])
            valid = logits.masked_fill(~mask, -torch.inf)
            probabilities = torch.where((tasks == 1)[:, None], valid.softmax(-1), logits.sigmoid()) * mask
        return VideoJevOutput(logits=logits, probabilities=probabilities, score_mask=mask)

    @torch.inference_mode()
    def predict_video(self, processor, video, questions, *, fps=2., max_frames=2048, context=262144, threshold=None):
        """Return ordered QA answers / grounding intervals from an original video file."""
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError('threshold must be between 0 and 1')
        pixels, record = prepare_video(processor, video, fps=fps, max_frames=max_frames, context=context)
        features = self.encode_video(pixels, torch.tensor([record['grid_thw']]))
        return self.predict_features(processor, features, record, questions, context=context, threshold=threshold)

    @torch.inference_mode()
    def predict_features(self, processor, features, record, questions, *, context=262144, threshold=None):
        """Use the configured grounding threshold unless this call overrides it."""
        decoder = dict(self.config.decoder)
        if threshold is not None:
            decoder['threshold'] = threshold
        if not 0 < decoder['threshold'] < 1:
            raise ValueError('threshold must be between 0 and 1')
        results = []
        for batch, rows in prepare_query_groups(processor, record, questions, context):
            output = self(**batch, visual_features=features)
            results.extend(format_predictions(output, rows, record, decoder))
        return results
