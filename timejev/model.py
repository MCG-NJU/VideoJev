"""Native Qwen3.5 modules and shared grounding/choice head."""
import contextlib
import math
import os
import torch
from torch import nn
from transformers import Qwen3_5Model
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
from .vision import VISUAL_PROTOCOL


class PointerHead(nn.Module):
    """One query/key similarity per native two-frame temporal group."""
    def __init__(self, hidden, dimension=256, head_norm='none', *,
                 similarity='dot_product', logit_scale=10.0):
        super().__init__()
        if head_norm not in ('none', 'layernorm'):
            raise ValueError('Unknown pointer head normalization')
        if similarity not in ('dot_product', 'cosine'):
            raise ValueError('Unknown pointer head similarity')
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError('Pointer head logit scale must be finite and positive')
        self.similarity = similarity
        self.head_norm = head_norm
        self.q = nn.Linear(hidden, dimension)
        self.k = nn.Linear(hidden, dimension)
        self.bias = nn.Parameter(torch.zeros(()))
        self.dimension = dimension
        self.scale = dimension ** -.5 if similarity == 'dot_product' else float(logit_scale)
        nn.init.normal_(self.q.weight, std=.005)
        nn.init.normal_(self.k.weight, std=.005)
        nn.init.zeros_(self.q.bias)
        nn.init.zeros_(self.k.bias)
        # Keep the historical initialization/RNG order and state keys unchanged
        # when disabled. Each branch has its own learnable affine parameters.
        if head_norm == 'layernorm':
            self.q_norm = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=True)

    def forward(self, pairs, query):
        # FP32 scoring, including under AMP.
        if self.head_norm == 'layernorm':
            keys = self.k(self.k_norm(pairs.float()))
            q = self.q(self.q_norm(query.float()))
        else:
            keys = self.k(pairs.float())
            q = self.q(query.float())
        if self.similarity == 'cosine':
            # Normalize the projected vectors, not the incoming LLM features.
            # The scale is fixed; the existing scalar bias remains trainable.
            keys = nn.functional.normalize(keys, dim=-1, eps=1e-6)
            q = nn.functional.normalize(q, dim=-1, eps=1e-6)
        return torch.einsum('bgd,bd->bg', keys, q) * self.scale + self.bias


class TimeJev(nn.Module):
    visual_protocol = VISUAL_PROTOCOL

    def __init__(self, base, head_dim=256, checkpointing=True, attention='flash_attention_2', config=None,
                 freeze_visual=False, query_placement='both', attention_mode='causal',
                 checkpoint_segment_size=0, checkpoint_segment_threshold=131072, feature_root=None,
                 feature_variant_root=None, head_norm='none', readout_mode='final_hidden', qk_init='pretrained',
                 head_similarity='dot_product', head_logit_scale=10.0):
        super().__init__()
        if head_norm not in ('none', 'layernorm'):
            raise ValueError('Unknown pointer head normalization')
        self.head_norm = head_norm
        if head_similarity not in ('dot_product', 'cosine'):
            raise ValueError('Unknown pointer head similarity')
        if not math.isfinite(head_logit_scale) or head_logit_scale <= 0:
            raise ValueError('Pointer head logit scale must be finite and positive')
        self.head_similarity, self.head_logit_scale = head_similarity, float(head_logit_scale)
        if readout_mode not in ('final_hidden', 'last_attention'):
            raise ValueError('Unknown readout mode')
        if qk_init != 'pretrained':
            raise ValueError('Only pretrained native attention weights are supported')
        self.readout_mode, self.qk_init = readout_mode, qk_init
        self.readout_metadata = None
        # FLA's optional intra-card CP backend is inference-only and showed a
        # large long-input mismatch. Its standard varlen CUDA kernels retain
        # exact recurrence and all packed sequence boundaries.
        os.environ['FLA_INTRACARD_CP']='0'
        # FP32 parameter/master storage with BF16 autocast preserves Qwen's FP32
        # sensitive parameters; no blanket downcast from DeepSpeed BF16 mode.
        if config is None:
            self.backbone = Qwen3_5Model.from_pretrained(
                base, dtype=torch.float32, attn_implementation=attention)
        else:
            config.dtype = torch.float32
            config.text_config.dtype = torch.float32
            config.vision_config.dtype = torch.float32
            config._attn_implementation = attention
            self.backbone = Qwen3_5Model(config)
        if query_placement not in ('both', 'last') or attention_mode != 'causal':
            raise ValueError('Unknown query placement or attention mode')
        self.freeze_visual = bool(freeze_visual)
        self.feature_root = feature_root
        self.feature_cache_id = None
        self.feature_cache_ids = set()
        if feature_root:
            if not self.freeze_visual:
                raise ValueError('Cached vision features require frozen encoder and merger')
            from .features import validate_cache, state_digest
            self.feature_spec = validate_cache(feature_root, [], 2., require_complete=False)
            if state_digest(self.backbone.visual.state_dict()) != self.feature_spec['vision_sha256']:
                raise ValueError('Visual weights differ from feature-cache weights')
            self.feature_cache_id = self.feature_spec['cache_id']
            self.feature_cache_ids.add(self.feature_cache_id)
            if feature_variant_root:
                variant_spec=validate_cache(feature_variant_root, [], 2., require_complete=False)
                if variant_spec['vision_sha256'] != self.feature_spec['vision_sha256']:
                    raise ValueError('Adaptive features use different visual weights')
                self.feature_cache_ids.add(variant_spec['cache_id'])
        self.query_placement, self.attention_mode = query_placement, attention_mode
        if attention_mode != 'causal' or checkpoint_segment_size:
            from .text_backbone import configure_text_backbone
            configure_text_backbone(self.backbone, attention_mode, checkpoint_segment_size, checkpoint_segment_threshold)
        if self.freeze_visual:
            self.backbone.visual.requires_grad_(False)
            self.backbone.visual.eval()
        self.backbone.config.use_cache = False
        if checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            # Nested activation recomputation keeps vision-layer inputs from staying
            # resident throughout the long language forward/backward. All native
            # operations and every input token are unchanged.
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
            if not self.freeze_visual:
                self.backbone.visual = checkpoint_wrapper(self.backbone.visual)
            else:
                self.backbone.visual.gradient_checkpointing_disable()
        replacement_head = None
        if readout_mode != 'final_hidden':
            from .readout import adapt_readout
            replacement_head, self.readout_metadata = adapt_readout(self.backbone, readout_mode, qk_init)
        self.head = (replacement_head if replacement_head is not None else
                     PointerHead(self.backbone.config.text_config.hidden_size, head_dim, head_norm=head_norm,
                                 similarity=head_similarity, logit_scale=head_logit_scale))
        self._packed_kernels_available=(is_fast_path_available and
            self.backbone.config.text_config._attn_implementation in ('flash_attention_2','flash_attention_3'))
        assert all(p.requires_grad for p in self.backbone.language_model.parameters())
        assert all(p.requires_grad != self.freeze_visual for p in self.backbone.visual.parameters())

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.backbone.visual.eval()
        return self

    def forward(self, input_ids, attention_mask, mm_token_type_ids,
                video_grid_thw, pair_indices, query_indices, score_mask, pixel_values_videos=None,
                visual_features=None, feature_cache_ids=None, labels=None,
                packed_sample_lengths=None, cu_seq_lens_q=None, cu_seq_lens_k=None,
                max_length_q=None, max_length_k=None, seq_idx=None, decision_tasks=None, **unused):
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if input_ids.is_cuda else contextlib.nullcontext()
        with amp:
            if packed_sample_lengths is not None:
                if not self._packed_kernels_available:
                    raise RuntimeError('Isolated packing requires FlashAttention, native FLA and causal-conv1d; fallback paths are not permitted')
                if (not self.freeze_visual or visual_features is None or pixel_values_videos is not None
                        or input_ids.shape[0]!=1 or attention_mask is not None
                        or len(feature_cache_ids)!=len(packed_sample_lengths)
                        or not all(x in self.feature_cache_ids for x in feature_cache_ids)):
                    raise ValueError('Packing requires verified frozen features and padding-free attention')
                if cu_seq_lens_q is None or cu_seq_lens_k is None or seq_idx is None:
                    raise ValueError('Missing FA/DeltaNet/convolution isolation boundaries')
                embeds=self.backbone.get_input_embeddings()(input_ids)
                features=visual_features.to(device=embeds.device,dtype=embeds.dtype)
                _,video_mask=self.backbone.get_placeholder_mask(input_ids,inputs_embeds=embeds,video_features=features)
                embeds=embeds.masked_scatter(video_mask,features)
                positions=[]; offset=0
                for i,n in enumerate(packed_sample_lengths):
                    positions.append(self.backbone.compute_3d_position_ids(
                        input_ids=input_ids[:,offset:offset+n],inputs_embeds=embeds[:,offset:offset+n],
                        video_grid_thw=video_grid_thw[i:i+1],attention_mask=None,
                        mm_token_type_ids=mm_token_type_ids[:,offset:offset+n]))
                    offset+=n
                if offset<input_ids.shape[1]:
                    positions.append(torch.arange(input_ids.shape[1]-offset,device=embeds.device)[None,None,:].expand(3,1,-1))
                positions=torch.cat(positions,dim=-1)
                language_output=self.backbone.language_model(inputs_embeds=embeds,position_ids=positions,
                    attention_mask={'full_attention':None,'linear_attention':None},use_cache=False,
                    cu_seq_lens_q=cu_seq_lens_q,cu_seq_lens_k=cu_seq_lens_k,
                    max_length_q=max_length_q,max_length_k=max_length_k,seq_idx=seq_idx)
                hidden = language_output.last_hidden_state
            elif visual_features is None:
                hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids, pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw, use_cache=False).last_hidden_state
            else:
                if (not self.freeze_visual or pixel_values_videos is not None or not self.feature_cache_id
                        or len(feature_cache_ids)!=input_ids.shape[0]
                        or not all(x in self.feature_cache_ids for x in feature_cache_ids)):
                    raise ValueError('Unverified/incompatible visual features')
                embeds = self.backbone.get_input_embeddings()(input_ids)
                features = visual_features.to(device=embeds.device, dtype=embeds.dtype)
                _, mask = self.backbone.get_placeholder_mask(input_ids, inputs_embeds=embeds, video_features=features)
                embeds = embeds.masked_scatter(mask, features)
                positions = self.backbone.compute_3d_position_ids(input_ids=input_ids, inputs_embeds=embeds,
                    video_grid_thw=video_grid_thw, attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids)
                hidden = self.backbone(inputs_embeds=embeds, position_ids=positions, attention_mask=attention_mask,
                    video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids, use_cache=False).last_hidden_state
        b = (torch.zeros(len(query_indices),device=hidden.device,dtype=torch.long) if packed_sample_lengths is not None
             else torch.arange(hidden.shape[0], device=hidden.device))
        pairs = hidden[b[:, None], pair_indices]
        query = hidden[b, query_indices]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = self.head(pairs, query)
            assert logits.shape == score_mask.shape
            loss = None
            if labels is not None:
                raw = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                loss = ((raw * score_mask).sum(1) / score_mask.sum(1).clamp_min(1)).mean()
        return dict(logits=logits, loss=loss)

    def load_checkpoint(self, checkpoint):
        if checkpoint.get('config', {}).get('formulation') != getattr(self, 'formulation', None):
            raise ValueError('Checkpoint decision formulation differs from this model; use its matching model factory')
        if checkpoint.get('visual_protocol') != self.visual_protocol:
            raise ValueError('Checkpoint must use the single-key pair-midpoint protocol; old dual-slot weights are incompatible')
        spec = checkpoint.get('config', {})
        if spec.get('experiment_arch', 'default') != getattr(self, 'experiment_arch', 'default'):
            raise ValueError('Checkpoint experimental architecture mismatch')
        if spec.get('attention_mode', 'causal') != self.attention_mode or spec.get('query_placement', 'both') != self.query_placement:
            raise ValueError('Checkpoint query placement/attention architecture mismatch')
        if spec.get('head_norm', 'none') != getattr(self, 'head_norm', 'none'):
            raise ValueError('Checkpoint pointer head normalization mismatch')
        if (spec.get('head_similarity', 'dot_product') != getattr(self, 'head_similarity', 'dot_product') or
                (spec.get('head_similarity') == 'cosine' and
                 spec.get('head_logit_scale', 10.0) != self.head_logit_scale)):
            raise ValueError('Checkpoint pointer head similarity or scale mismatch')
        if (spec.get('readout_mode', 'final_hidden') != getattr(self, 'readout_mode', 'final_hidden') or
                spec.get('qk_init', 'pretrained') != getattr(self, 'qk_init', 'pretrained')):
            raise ValueError('Checkpoint readout architecture or Q/K initialization mismatch')
        self.load_state_dict(checkpoint['model'], strict=True)
        if self.feature_cache_id:
            from .features import state_digest
            if state_digest(self.backbone.visual.state_dict()) != self.feature_spec['vision_sha256']:
                raise ValueError('Loaded checkpoint visual weights differ from cache')


def from_config(config, *, checkpointing=None):
    return TimeJev(config['base'], config['head_dim'],
                   config.get('checkpointing', True) if checkpointing is None else checkpointing,
                   freeze_visual=config.get('freeze_visual', False),
                   query_placement=config.get('query_placement', 'both'),
                   attention_mode=config.get('attention_mode', 'causal'),
                   checkpoint_segment_size=config.get('checkpoint_segment_size', 0),
                   checkpoint_segment_threshold=config.get('checkpoint_segment_threshold', 131072),
                   feature_root=config.get('feature_root'),
                   feature_variant_root=config.get('feature_variant_root'),
                   head_norm=config.get('head_norm', 'none'),
                   head_similarity=config.get('head_similarity', 'dot_product'),
                   head_logit_scale=config.get('head_logit_scale', 10.0),
                   readout_mode=config.get('readout_mode', 'final_hidden'),
                   qk_init=config.get('qk_init', 'pretrained'))
