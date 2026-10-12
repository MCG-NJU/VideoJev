"""Per-question VTG and MCQ losses for the shared cosine decision head."""
import math
import torch
from torch.nn import functional as F

from .decision_inputs import CHOICE
from .decision_model import typed_outputs, _duration_soft_iou_loss


def decision_losses(logits, score_mask, decision_tasks, labels=None, *,
                    score_durations=None, soft_iou_weight=0.):
    for weight in (soft_iou_weight,):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    output = typed_outputs(logits, score_mask, decision_tasks)
    if labels is None:
        return output
    if not ((labels[score_mask] == 0) | (labels[score_mask] == 1)).all():
        raise ValueError('Expected binary midpoint or one-hot choice targets')
    valid = score_mask.any(1)
    choice = decision_tasks == CHOICE
    if (choice & valid & ((labels * score_mask).sum(1) != 1)).any():
        raise ValueError('Single-choice targets must contain exactly one correct option')
    values = logits.float().masked_fill(~score_mask, 0.)
    targets = labels.float().masked_fill(~score_mask, 0.)
    bce = (F.binary_cross_entropy_with_logits(values, targets, reduction='none') * score_mask).sum(1)
    bce = bce / score_mask.sum(1).clamp_min(1)
    safe = values.masked_fill(~score_mask, -torch.inf)
    safe = torch.where(valid[:, None], safe, torch.zeros_like(safe))
    ce = -(targets * safe.log_softmax(-1).masked_fill(~score_mask, 0.)).sum(1)
    iou = (_duration_soft_iou_loss(output['probabilities'], labels, score_mask,
                                 decision_tasks, score_durations)
           if soft_iou_weight else bce * 0.)
    per_query = torch.where(choice, ce, bce) + soft_iou_weight * iou
    output['loss'] = (per_query * valid).sum() / valid.sum().clamp_min(1)
    # Sums and counts allow the trainer to aggregate by real question count.
    output['loss_sums'] = torch.stack(((bce * valid * ~choice).sum(),
        (ce * valid * choice).sum(), (iou * valid).sum())).detach()
    output['task_counts'] = torch.stack(((valid & ~choice).sum(), (valid & choice).sum())).detach()
    return output
