"""One shared pointer head, with Bernoulli grounding and categorical choice outputs."""
import math
import torch
from torch.nn import functional as F
from .model import TimeJev
from .decision_inputs import FORMULATION, SPECIAL_TOKENS, CHOICE, load_processor, token_metadata


def _duration_soft_iou_loss(probabilities, labels, score_mask, decision_tasks, score_durations):
    """Per-query soft IoU loss; no-positive grounding and QA add exactly zero."""
    grounding_mask = score_mask & (decision_tasks != CHOICE)[:, None]
    if not grounding_mask.any():
        return probabilities.new_zeros(probabilities.shape[0])
    if score_durations is None or score_durations.shape != probabilities.shape:
        raise ValueError('Soft IoU requires real score-cell durations matching logits')
    if (not torch.isfinite(score_durations[grounding_mask]).all() or
            (score_durations[grounding_mask] <= 0).any()):
        raise ValueError('Grounding score-cell durations must be finite and positive')
    durations = torch.where(grounding_mask, score_durations.to(probabilities.dtype), 0.)
    targets = torch.where(grounding_mask, labels.float(), 0.)
    intersection = (durations * probabilities * targets).sum(1)
    union = (durations * (targets + probabilities * (1. - targets))).sum(1)
    positive = (durations * targets).sum(1) > 0
    # Positive GT guarantees positive union, without an epsilon changing the ratio.
    # No positive midpoint (including empty packs) keeps the original BCE only.
    denominator = torch.where(positive, union, torch.ones_like(union))
    return torch.where(positive, 1. - intersection / denominator, 0.)


def typed_outputs(logits, score_mask, decision_tasks, labels=None, *,
                  score_durations=None, soft_iou_weight=0.0):
    valid = score_mask.any(1)
    choice = decision_tasks == CHOICE
    if ((decision_tasks != 0) & ~choice).any():
        raise ValueError('Unknown decision task index')
    # Avoid all-masked softmax for zero-loss scheduling placeholders.
    safe = logits.masked_fill(~score_mask, -torch.inf)
    safe = torch.where(valid[:, None], safe, torch.zeros_like(safe))
    probabilities = torch.where(choice[:, None], safe.softmax(-1), logits.sigmoid()) * score_mask
    loss = None
    if labels is not None:
        if (labels[score_mask] < 0).any() or (labels[score_mask] > 1).any():
            raise ValueError('Decision targets must be probabilities in [0,1]')
        if (choice & valid & (labels.sum(1) != 1)).any():
            raise ValueError('Single-choice targets must sum to one')
        bce = (F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none') * score_mask).sum(1)
        bce = bce / score_mask.sum(1).clamp_min(1)
        log_probs = safe.log_softmax(-1).masked_fill(~score_mask, 0.)
        ce = -(labels * log_probs).sum(1)
        per_sample = torch.where(choice, ce, bce)
        if soft_iou_weight:
            if not math.isfinite(soft_iou_weight) or soft_iou_weight < 0:
                raise ValueError('Soft IoU weight must be finite and nonnegative')
            per_sample = per_sample + soft_iou_weight * _duration_soft_iou_loss(
                probabilities, labels, score_mask, decision_tasks, score_durations)
        loss = (per_sample * valid).sum() / valid.sum().clamp_min(1)
    return dict(logits=logits, probabilities=probabilities, loss=loss)


class DecisionTimeJev(TimeJev):
    formulation = FORMULATION

    def __init__(self, base, *args, tokenizer=None, grounding_loss='bce', soft_iou_weight=0.0, **kwargs):
        if (kwargs.get('query_placement') != 'last' or
                kwargs.get('attention_mode') != 'causal'):
            raise ValueError('Decision v1 requires trailing question and a supported full-attention mode')
        if grounding_loss not in ('bce', 'bce_soft_iou'):
            raise ValueError('Unknown grounding loss')
        soft_iou_weight = float(soft_iou_weight)
        if not math.isfinite(soft_iou_weight) or soft_iou_weight < 0:
            raise ValueError('Soft IoU weight must be finite and nonnegative')
        if grounding_loss == 'bce' and soft_iou_weight != 0:
            raise ValueError('Nonzero Soft IoU weight requires grounding_loss=bce_soft_iou')
        super().__init__(base, *args, **kwargs)
        self.grounding_loss, self.soft_iou_weight = grounding_loss, soft_iou_weight
        tokenizer = tokenizer or load_processor(base).tokenizer
        # Attention changes model behavior, not the prompt/token-length identity.
        # Keep decision_inputs and its existing length-cache fingerprint unchanged.
        self.decision_metadata = token_metadata(tokenizer) | {'attention_mode': self.attention_mode}
        if self.head_similarity == 'cosine':
            self.decision_metadata['head_scoring'] = dict(similarity='cosine',
                logit_scale=self.head_logit_scale, trainable_scale=False,
                normalization='L2 after Q/K projection', eps=1e-6, trainable_bias=True)
        if getattr(self, 'readout_metadata', None) is not None:
            self.decision_metadata['readout'] = self.readout_metadata
        embedding = self.backbone.get_input_embeddings()
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        initial = embedding.weight[im_end].detach().clone()
        # Some Qwen checkpoints have padded vocabulary rows. Never shrink their
        # original vocabulary or overwrite an existing tokenizer token.
        if len(tokenizer) > embedding.num_embeddings:
            self.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        embedding = self.backbone.get_input_embeddings()
        with torch.no_grad():
            for token in SPECIAL_TOKENS:
                embedding.weight[tokenizer.convert_tokens_to_ids(token)].copy_(initial)

    def forward(self, *, decision_tasks, score_mask, labels=None, score_durations=None, **kwargs):
        out = super().forward(score_mask=score_mask, labels=None, decision_tasks=decision_tasks, **kwargs)
        return typed_outputs(out['logits'], score_mask, decision_tasks, labels,
                             score_durations=score_durations, soft_iou_weight=self.soft_iou_weight)

    def load_checkpoint(self, checkpoint):
        if checkpoint.get('config', {}).get('formulation') != FORMULATION:
            raise ValueError('Legacy and decision checkpoints are not interchangeable')
        if checkpoint.get('decision_metadata') != self.decision_metadata:
            raise ValueError('Checkpoint decision token IDs, readout protocol or attention mode differ')
        super().load_checkpoint(checkpoint)


def from_config(config, *, checkpointing=None):
    from .config import normalize_recipe
    config = normalize_recipe(config)
    if config.get('formulation') != FORMULATION or config.get('shared_decision_head') is not True:
        raise ValueError('Expected decision v1 with one shared grounding/QA head')
    from .multiturn_model import MultiturnTimeJev
    if (config.get('experiment_arch') != 'timestamp_option_memory'
            or config.get('grounding_candidates') != 'timestamp_options'
            or config.get('training_layout') != 'shared_video_multiturn_v1'
            or config.get('attention_mode') != 'causal'
            or config.get('readout_mode') != 'last_attention'):
        raise ValueError('Video-Jev supports the final causal timestamp-option architecture')
    model_class = MultiturnTimeJev
    if config.get('input_mode') == 'rgb_dynamic':
        from .rgb_model import RGBMultiturnTimeJev
        model_class = RGBMultiturnTimeJev
    architecture_kwargs = dict(grounding_candidates='timestamp_options',
        freeze_llm=config.get('freeze_llm', False))
    return model_class(config['base'], config['head_dim'],
        config.get('checkpointing', True) if checkpointing is None else checkpointing,
        freeze_visual=config.get('freeze_visual', True), query_placement=config['query_placement'],
        attention_mode=config['attention_mode'], checkpoint_segment_size=config.get('checkpoint_segment_size',0),
        checkpoint_segment_threshold=config.get('checkpoint_segment_threshold',131072),
        feature_root=config.get('feature_root'), feature_variant_root=config.get('feature_variant_root'),
        head_norm=config.get('head_norm', 'none'), grounding_loss=config.get('grounding_loss', 'bce'),
        head_similarity=config.get('head_similarity', 'dot_product'),
        head_logit_scale=config.get('head_logit_scale', 10.0),
        soft_iou_weight=config.get('soft_iou_weight', 0.0),
        readout_mode=config.get('readout_mode', 'final_hidden'), qk_init=config.get('qk_init', 'pretrained'),
        **architecture_kwargs)
