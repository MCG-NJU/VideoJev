"""Head warmup handoff; each stage uses a fresh AdamW and its own LR schedule."""
import math
from pathlib import Path

import torch

from .engine import sha256


def optimizer_groups(model, config):
    groups = []
    for head in (False, True):
        for decay in (True, False):
            params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith('head.') == head and (p.ndim >= 2) == decay]
            # ZeRO must not receive empty LLM groups during head-only warmup.
            if params:
                groups.append(dict(params=params, lr=config['head_lr'] if head else config['lr'],
                                   weight_decay=config['weight_decay'] if decay else 0.))
    return groups


def lr_factor(index, total_steps, warmup_ratio):
    warmup = max(1, round(total_steps * warmup_ratio))
    return ((index + 1) / warmup if index < warmup else
            .1 + .9 * .5 * (1 + math.cos(math.pi * min(1,
                (index - warmup) / max(1, total_steps - warmup)))))


def initial_head(model, path, config):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = checkpoint['config']
    if (checkpoint.get('format') != 'timejev_head_stage_v1'
            or source.get('training_stage') != 'head_only'
            or config.get('training_stage') != 'joint'):
        raise ValueError('Expected a head-only checkpoint for the joint stage')
    for key in ('base', 'training_split_report', 'seed', 'formulation', 'experiment_arch',
                'grounding_candidates', 'readout_mode', 'attention_mode', 'head_dim',
                'head_similarity', 'head_logit_scale', 'global_batch_size'):
        if source.get(key) != config.get(key):
            raise ValueError(f'Head handoff differs in {key}')
    for key in ('input_mode', 'max_sequence_length', 'rgb_metadata'):
        if source.get(key) != config.get(key):
            raise ValueError(f'Head input protocol differs in {key}')
    def architecture_metadata(value):
        return {k: v for k, v in value.items() if k not in ('timestamp_loss', 'timestamp_ablation')}
    if architecture_metadata(checkpoint['decision_metadata']) != architecture_metadata(model.decision_metadata):
        raise ValueError('Head readout/token/scoring metadata differ')
    if checkpoint['split_sha256'] != sha256(config['training_split_report']):
        raise ValueError('Head checkpoint belongs to another video split')
    model.head.load_state_dict(checkpoint['head'], strict=True)
    return dict(checkpoint=str(Path(path).resolve()), sha256=sha256(path),
                source_run=checkpoint['source_run'], steps=checkpoint['steps'],
                head_peak_lr=source['head_lr'],
                head_training_fraction=source['training_fraction'],
                training_queries=checkpoint['training_queries'],
                training_seconds=checkpoint['training_seconds'], optimizer_reset=True,
                scheduler_reset=True, backbone_source=config['base'],
                reused_head_stage=False)
