"""Native causal Qwen execution with isolated complete-video/query sequences."""
import contextlib
import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from transformers.integrations.flash_attention import get_target_dtype
from transformers.modeling_flash_attention_utils import fa_peft_integration_check
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

from .decision_inputs import CHOICE, load_processor
from .decision_model import DecisionTimeJev, typed_outputs
from .memory_reader import MemoryReaderTimeJev


def _varlen_attention(query, key, value, query_lengths, key_lengths, scale, dropout=0., *, causal=False):
    """Native GQA FA on CUDA; tiny CPU formula checks use independent SDPA."""
    if not query_lengths:
        return query
    if query.is_cuda:
        from flash_attn import flash_attn_varlen_func
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError('Asymmetric FlashAttention requires native BF16/FP16 activations')
        qlens = torch.tensor(query_lengths, dtype=torch.int32, device=query.device)
        klens = torch.tensor(key_lengths, dtype=torch.int32, device=query.device)
        cu_q = torch.cat((qlens.new_zeros(1), qlens.cumsum(0, dtype=torch.int32)))
        cu_k = torch.cat((klens.new_zeros(1), klens.cumsum(0, dtype=torch.int32)))
        return flash_attn_varlen_func(query, key, value, cu_q, cu_k,
            max(query_lengths), max(key_lengths), dropout_p=dropout,
            softmax_scale=scale, causal=causal)
    output, qstart, kstart = [], 0, 0
    for qlength, klength in zip(query_lengths, key_lengths):
        q = query[qstart:qstart+qlength].transpose(0, 1)[None]
        k = key[kstart:kstart+klength].transpose(0, 1)[None]
        v = value[kstart:kstart+klength].transpose(0, 1)[None]
        # FA2 aligns a cached suffix at the bottom-right of the KV sequence.
        # SDPA is_causal=True uses top-left alignment for unequal Q/K lengths.
        mask = None
        if causal:
            qpos = torch.arange(qlength, device=q.device) + klength - qlength
            mask = torch.arange(klength, device=q.device)[None] <= qpos[:, None]
        output.append(F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout,
            is_causal=False, scale=scale, enable_gqa=True)[0].transpose(0, 1))
        qstart += qlength; kstart += klength
    return torch.cat(output)


class AsymmetricQwenAttention(Qwen3_5Attention):
    """Original Qwen projections/norms/rotary/gate/output; visibility only changes."""
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, asymmetric_lengths=None, asymmetric_prefix_lengths=None, **kwargs):
        if attention_mask is not None or hidden_states.shape[0] != 1:
            raise ValueError('Asymmetric attention requires an explicitly unpadded token stream')
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim*2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
        continuation = past_key_values is not None and getattr(past_key_values, '_timejev_prefix_continuation', False)
        if continuation:
            # These complete-query continuations never mutate or retain enlarged
            # attention KV. Only the much smaller DeltaNet states are copied.
            cached = past_key_values.layers[self.layer_idx]
            key_states = torch.cat((cached.keys, key_states), dim=-2)
            value_states = torch.cat((cached.values, value_states), dim=-2)
        elif past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        q, k, v = (x[0].transpose(0, 1).contiguous() for x in (query_states, key_states, value_states))
        if q.is_cuda:
            # Exactly the installed Qwen FA wrapper's dtype restoration, after
            # native QK norms/RoPE and after storing the unmodified native KV.
            q, k, v = fa_peft_integration_check(q, k, v, get_target_dtype(q, self))
        dropout = self.attention_dropout if self.training else 0.
        if continuation:
            output = _varlen_attention(q, k, v, [len(q)], [len(k)], self.scaling, dropout, causal=True)
        else:
            output = _varlen_attention(q, k, v, asymmetric_lengths, asymmetric_lengths,
                self.scaling, dropout, causal=True)
        output = output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output * torch.sigmoid(gate)), None


def _native_tensors(cache):
    for layer in cache.layers:
        if hasattr(layer, 'keys'):
            yield layer.keys
            yield layer.values
        for field in ('conv_states', 'recurrent_states'):
            if hasattr(layer, field):
                yield from getattr(layer, field).values()


def _tensor_version(tensor):
    return (id(tensor), tensor._version, tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device)


def _fork_prefix_cache(original):
    """Share immutable KV and privately clone every mutable native Delta state."""
    result = copy.copy(original)
    result.layers = []
    for layer in original.layers:
        duplicate = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, dict):
                setattr(duplicate, name, {key: (item.clone() if torch.is_tensor(item) else item)
                    for key, item in value.items()})
        result.layers.append(duplicate)
    result._timejev_prefix_continuation = True
    return result


@dataclass(frozen=True)
class AsymmetricVideoMemory:
    temporal_hidden: torch.Tensor
    native_cache: DynamicCache
    prefix_ids: torch.Tensor
    prefix_types: torch.Tensor
    prefix_positions: torch.Tensor
    grid: torch.Tensor
    temporal_indices: torch.Tensor
    video_identity: object
    model_identity: int
    parameter_versions: tuple
    feature_cache_id: object
    tensor_versions: tuple
    storage_bytes: int

    @property
    def prefix_length(self):
        return len(self.prefix_ids)


class AsymmetricVideoMemoryTimeJev(MemoryReaderTimeJev):
    experiment_arch = 'asymmetric_video_memory'

    def __init__(self, base, *args, **kwargs):
        mode = kwargs.get('attention_mode')
        if (mode != 'causal' or self.experiment_arch != 'timestamp_option_memory'
                or kwargs.get('readout_mode') != 'last_attention'
                or not kwargs.get('freeze_visual', False)):
            raise ValueError('Asymmetric memory requires frozen vision and the skip-last-MLP baseline')
        tokenizer = kwargs.get('tokenizer')
        if tokenizer is None:
            tokenizer = load_processor(base).tokenizer
            kwargs['tokenizer'] = tokenizer
        # Reuse the baseline's exact loading/initialization and native layer
        # checkpoint configuration, then replace only ordinary FA's forward.
        kwargs['attention_mode'] = 'causal'
        DecisionTimeJev.__init__(self, base, *args, **kwargs)
        self.attention_mode = mode
        self.backbone.language_model._timejev_attention_mode = self.attention_mode
        for layer in self.backbone.language_model.layers:
            if layer.block_type == 'full_attention':
                layer.self_attn.__class__ = AsymmetricQwenAttention
                layer.self_attn._timejev_fully_causal = mode == 'causal'
        self.vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.video_token_id = tokenizer.convert_tokens_to_ids('<|video_pad|>')
        self._active_video_cache = None
        self._video_cache_generation = 0
        self.last_video_cache_stats = {}
        self.decision_metadata['attention_mode'] = self.attention_mode
        self.decision_metadata['architecture'] = dict(experiment_arch=self.experiment_arch,
            full_attention='all tokens causal within each complete sample',
            deltanet='native causal video-to-suffix recurrence; reset only between samples',
            split='after_last_native_vision_end', full_video_tokens=True,
            positions='original native MRoPE', head='original shared PointerHead', final_mlp=False,
            cache='native per-layer KV plus convolution/recurrent states plus final temporal hidden',
            cache_dtype='native unmodified', additional_trainable_parameters=False)

    def train(self, mode=True):
        if mode:
            self._video_cache_generation = getattr(self, '_video_cache_generation', 0)+1
        return super().train(mode)

    def _versions(self):
        return (('training_generation', self._video_cache_generation),
            *(_tensor_version(parameter) for parameter in self.backbone.parameters()))

    def _run_layers(self, embeds, positions, lengths, prefix_lengths, native_cache=None):
        if (embeds.shape[0] != 1 or positions.shape[-1] != embeds.shape[1]
                or sum(lengths) != embeds.shape[1] or any(length <= 0 for length in lengths)
                or len(lengths) != len(prefix_lengths)):
            raise ValueError('Asymmetric stacks require complete, unpadded sample boundaries')
        if native_cache is not None and (self.training or torch.is_grad_enabled() or len(lengths) != 1):
            raise RuntimeError('Persistent native states are inference-only and belong to one video')
        text = self.backbone.language_model
        text_positions = None
        if positions.ndim == 3 and positions.shape[0] == 4:
            text_positions, positions = positions[0], positions[1:]
        if positions.ndim != 3 or positions.shape[0] != 3:
            raise ValueError('Expected original Qwen MRoPE positions')
        arguments = dict(attention_mask=None, position_ids=text_positions,
            past_key_values=native_cache, use_cache=native_cache is not None,
            asymmetric_lengths=lengths, asymmetric_prefix_lengths=prefix_lengths)
        if native_cache is None:
            # Critically: one Delta recurrence/conv sequence per full sample,
            # never one sequence per video and suffix.
            lens = torch.tensor(lengths, device=embeds.device, dtype=torch.int32)
            cu = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
            seq_idx = torch.repeat_interleave(torch.arange(len(lengths), device=embeds.device,
                dtype=torch.int32), lens.long())[None]
            arguments.update(cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                max_length_q=max(lengths), max_length_k=max(lengths), seq_idx=seq_idx)
        segment_size = getattr(text, '_timejev_segment_size', 0)
        segmented = (self.training and torch.is_grad_enabled() and segment_size > 0
            and embeds.shape[1] > text._timejev_segment_threshold)
        width = segment_size if segmented else 1
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if embeds.is_cuda else contextlib.nullcontext()
        with amp:
            rotary = text.rotary_emb(embeds, positions)
            hidden = embeds
            for offset in range(0, len(text.layers), width):
                segment = tuple(text.layers[offset:offset+width])
                def run(value, segment=segment):
                    for layer in segment:
                        value = layer(value, position_embeddings=rotary, **arguments)
                    return value
                hidden = checkpoint(run, hidden, use_reentrant=False) if segmented else run(hidden)
            hidden = text.norm(hidden)
        return list(hidden[0].split(lengths, dim=0))

    def _embeddings(self, samples, visual_features, feature_cache_ids, *, prefix_only=False):
        if (not self._packed_kernels_available or visual_features is None or feature_cache_ids is None
                or len(feature_cache_ids) != len(samples)
                or not all(key in self.feature_cache_ids for key in feature_cache_ids)):
            raise ValueError('Asymmetric execution requires native packed kernels and verified frozen features')
        pieces, offset = [], 0
        for sample in samples:
            ids = sample.ids[:sample.split] if prefix_only else sample.ids
            embeds = self.backbone.get_input_embeddings()(ids[None])
            count = int((ids == self.video_token_id).sum())
            features = visual_features[offset:offset+count].to(embeds)
            _, mask = self.backbone.get_placeholder_mask(ids[None], inputs_embeds=embeds, video_features=features)
            pieces.append(embeds.masked_scatter(mask, features)); offset += count
        if offset != len(visual_features):
            raise ValueError('Every original native visual token must be supplied exactly once')
        return pieces

    def _score_hidden(self, samples, candidates, queries, score_mask):
        pairs = torch.stack([F.pad(value, (0, 0, 0, score_mask.shape[1]-len(value))) for value in candidates])
        query = torch.stack(queries)
        with torch.autocast(device_type=query.device.type, enabled=False):
            return self.head(pairs, query)

    def _empty_logits(self, score_mask):
        embedding = self.backbone.get_input_embeddings()
        ids = torch.full((1, 64), self.vision_end_id, device=embedding.weight.device, dtype=torch.long)
        positions = torch.arange(64, device=ids.device)[None, None].expand(3, 1, -1)
        hidden = self._run_layers(embedding(ids), positions, [64], [32])[0]
        with torch.autocast(device_type=ids.device.type, enabled=False):
            zero = self.head(hidden[31][None, None], hidden[63][None]).sum()*0.
        return zero.expand(score_mask.shape)

    def forward(self, *, visual_features=None, feature_cache_ids=None, pixel_values_videos=None,
                score_mask, decision_tasks, labels=None, score_durations=None, **kwargs):
        if pixel_values_videos is not None:
            raise ValueError('Asymmetric memory uses the verified complete native visual features')
        samples = self._samples(score_mask=score_mask, decision_tasks=decision_tasks, **kwargs)
        if not samples:
            logits = self._empty_logits(score_mask)
        elif self._active_video_cache is not None:
            memories = self._resolve_cached(samples, visual_features, feature_cache_ids)
            logits = self._cached_logits(samples, memories, score_mask)
        else:
            pieces = self._embeddings(samples, visual_features, feature_cache_ids)
            hidden = self._run_layers(torch.cat(pieces, dim=1),
                torch.cat([item.positions for item in samples], dim=-1),
                [len(item.ids) for item in samples], [item.split for item in samples])
            logits = self._score_hidden(samples, [value[item.pairs] for item, value in zip(samples, hidden)],
                [value[item.query] for item, value in zip(samples, hidden)], score_mask)
        return typed_outputs(logits, score_mask, decision_tasks, labels,
            score_durations=score_durations, soft_iou_weight=self.soft_iou_weight)

    @torch.inference_mode(False)
    @torch.no_grad()
    def _encode_video_sample(self, sample, features, cache_id, versions):
        embeds = self._embeddings([sample], features, [cache_id], prefix_only=True)[0]
        cache = DynamicCache(config=self.backbone.config.text_config)
        hidden = self._run_layers(embeds, sample.positions[:, :, :sample.split],
            [sample.split], [sample.split], native_cache=cache)[0]
        ends = ((sample.ids[:sample.split] == self.vision_end_id)
            & (torch.roll(sample.ids[:sample.split], 1) == self.video_token_id)).nonzero().flatten()
        temporal = hidden[ends].detach().clone()
        tensors = list(_native_tensors(cache))
        if any(value is None for value in tensors):
            raise RuntimeError('Native prefix cache has missing KV or DeltaNet state')
        storage = temporal.numel()*temporal.element_size()+sum(value.numel()*value.element_size() for value in tensors)
        return AsymmetricVideoMemory(temporal, cache, sample.ids[:sample.split].clone(),
            sample.types[:sample.split].clone(), sample.positions[:, :, :sample.split].clone(),
            sample.grid.clone(), ends.clone(), sample.video_identity, id(self), versions,
            cache_id, tuple(_tensor_version(value) for value in [temporal, *tensors]), storage)

    def _validate_memory(self, memory, item, versions, feature_cache_id=None):
        if not isinstance(memory, AsymmetricVideoMemory):
            raise ValueError('Expected native asymmetric video memory')
        if memory.model_identity != id(self) or memory.parameter_versions != versions:
            raise ValueError('Video memory belongs to another model or stale parameter version')
        if memory.video_identity != item.video_identity:
            raise ValueError('Video identity differs from cached prefix')
        if feature_cache_id is not None and feature_cache_id != memory.feature_cache_id:
            raise ValueError('Feature-cache identity differs from cached prefix')
        if (memory.temporal_hidden.device != item.ids.device or memory.prefix_length != item.split
                or not torch.equal(memory.prefix_ids, item.ids[:item.split])
                or not torch.equal(memory.prefix_types, item.types[:item.split])
                or not torch.equal(memory.prefix_positions, item.positions[:, :, :item.split])
                or not torch.equal(memory.grid, item.grid)):
            raise ValueError('Native video tokens, positions, grid or device differ from cache')
        if memory.tensor_versions != tuple(_tensor_version(value)
                for value in [memory.temporal_hidden, *_native_tensors(memory.native_cache)]):
            raise ValueError('Cached native prefix state was modified')

    def _cached_logits(self, samples, memories, score_mask):
        candidates, queries = [], []
        for item, memory in zip(samples, memories):
            embeds = self.backbone.get_input_embeddings()(item.ids[None, item.split:])
            hidden = self._run_layers(embeds, item.positions[:, :, item.split:],
                [len(item.ids)-item.split], [0], native_cache=_fork_prefix_cache(memory.native_cache))[0]
            candidates.append(hidden[item.pairs-item.split] if item.task == CHOICE else memory.temporal_hidden)
            queries.append(hidden[item.query-item.split])
        return self._score_hidden(samples, candidates, queries, score_mask)

    def encode_video(self, *, visual_features=None, feature_cache_ids=None, **batch):
        self._require_cache_mode()
        samples = self._samples(**batch)
        if not samples:
            return []
        if visual_features is None or feature_cache_ids is None or len(feature_cache_ids) != len(samples):
            raise ValueError('Encoding videos requires their complete visual features and cache identities')
        output, offset, versions = [], 0, self._versions()
        for item, cache_id in zip(samples, feature_cache_ids):
            count = int((item.ids[:item.split] == self.video_token_id).sum())
            output.append(self._encode_video_sample(item, visual_features[offset:offset+count], cache_id, versions))
            offset += count
        if offset != len(visual_features):
            raise ValueError('Native visual feature count differs from complete input count')
        return output

    def forward_with_video_memory(self, video_memory, *, score_mask, decision_tasks,
                                  labels=None, score_durations=None, **batch):
        self._require_cache_mode()
        samples = self._samples(score_mask=score_mask, decision_tasks=decision_tasks, **batch)
        if len(samples) != len(video_memory):
            raise ValueError('One native video memory is required per question')
        feature_ids, versions = batch.get('feature_cache_ids'), self._versions()
        if feature_ids is not None and len(feature_ids) != len(samples):
            raise ValueError('One feature-cache identity is required per question')
        for index, (memory, item) in enumerate(zip(video_memory, samples)):
            self._validate_memory(memory, item, versions, feature_ids[index] if feature_ids else None)
        logits = self._cached_logits(samples, video_memory, score_mask) if samples else self._empty_logits(score_mask)
        return typed_outputs(logits, score_mask, decision_tasks, labels,
            score_durations=score_durations, soft_iou_weight=self.soft_iou_weight)

    @contextlib.contextmanager
    def video_cache(self, max_tokens=147456):
        self._require_cache_mode()
        if type(max_tokens) is not int or max_tokens < 1 or self._active_video_cache is not None:
            raise ValueError('A positive native prefix-token budget and nonnested cache context are required')
        stats = dict(hits=0, misses=0, same_pack_hits=0, queries=0, video_encoded_tokens=0,
            suffix_encoded_tokens=0, saved_video_tokens=0, evictions=0, peak_cache_tokens=0,
            cache_tokens=0, cache_bytes=0, peak_cache_bytes=0,
            cache_contents='native_per_layer_kv_conv_recurrent_and_temporal_hidden')
        state = dict(entries=OrderedDict(), stats=stats, max_tokens=max_tokens, versions=self._versions())
        self._active_video_cache = state
        try:
            yield stats
        finally:
            state['entries'].clear()
            stats['cache_tokens'] = stats['cache_bytes'] = 0
            self.last_video_cache_stats = dict(stats)
            self._active_video_cache = None

    def _resolve_cached(self, samples, visual_features, feature_cache_ids):
        self._require_cache_mode()
        state, versions = self._active_video_cache, self._versions()
        if versions != state['versions']:
            raise ValueError('Model parameters changed inside native video-cache context')
        if (feature_cache_ids is None or len(feature_cache_ids) != len(samples)
                or not all(key in self.feature_cache_ids for key in feature_cache_ids)):
            raise ValueError('Cached inference requires one verified feature-cache identity per question')
        entries, stats, resolved, result, offset = state['entries'], state['stats'], {}, [], 0
        for item, cache_id in zip(samples, feature_cache_ids):
            count = int((item.ids[:item.split] == self.video_token_id).sum())
            features = visual_features[offset:offset+count] if visual_features is not None else None
            offset += count
            key = ((str(item.video_identity), cache_id, tuple(item.grid.tolist()), item.split)
                if item.video_identity is not None else None)
            stats['queries'] += 1
            stats['suffix_encoded_tokens'] += len(item.ids)-item.split
            if key is not None and key in resolved:
                memory = resolved[key]
                stats['hits'] += 1; stats['same_pack_hits'] += 1
                stats['saved_video_tokens'] += item.split
            elif key is not None and key in entries:
                memory = entries.pop(key); entries[key] = memory
                stats['hits'] += 1; stats['saved_video_tokens'] += item.split
            else:
                if features is None:
                    raise ValueError('An uncached video requires every native visual feature')
                memory = self._encode_video_sample(item, features, cache_id, versions)
                stats['misses'] += 1; stats['video_encoded_tokens'] += item.split
                if key is not None and item.split <= state['max_tokens']:
                    while stats['cache_tokens']+item.split > state['max_tokens']:
                        _, old = entries.popitem(last=False)
                        stats['cache_tokens'] -= old.prefix_length
                        stats['cache_bytes'] -= old.storage_bytes
                        stats['evictions'] += 1
                    entries[key] = memory
                    stats['cache_tokens'] += item.split
                    stats['cache_bytes'] += memory.storage_bytes
                    stats['peak_cache_tokens'] = max(stats['peak_cache_tokens'], stats['cache_tokens'])
                    stats['peak_cache_bytes'] = max(stats['peak_cache_bytes'], stats['cache_bytes'])
            self._validate_memory(memory, item, versions, cache_id)
            if key is not None:
                resolved[key] = memory
            result.append(memory)
        if visual_features is not None and offset != len(visual_features):
            raise ValueError('Cached inference visual feature count differs')
        return result

    def load_checkpoint(self, checkpoint_data):
        config = checkpoint_data.get('config', {})
        if (config.get('experiment_arch') != self.experiment_arch
                or config.get('attention_mode') != self.attention_mode):
            raise ValueError('Checkpoint asymmetric attention architecture differs')
        DecisionTimeJev.load_checkpoint(self, checkpoint_data)
