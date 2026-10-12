"""Timestamp candidates in the text suffix, with unchanged native video tokens.

Displayed timestamps are copied from the official video replacement. Their
list positions bind to full-precision score_times; rounded text never defines
labels or interval boundaries.
"""
import math
import re

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from .decision_inputs import (FORMULATION, DECIDE, OPTION_END, SPECIAL_TOKENS, GROUNDING,
    CHOICE, load_processor, task_of, validate_row, suffix as original_suffix,
    token_metadata as original_token_metadata)

GROUNDING_CANDIDATES = 'timestamp_options'
INPUT_PROTOCOL = 'native_timestamp_option_suffix_v1'
_TIMESTAMP = re.compile(r'<[0-9]+\.[0-9] seconds>')


def token_metadata(tokenizer):
    return original_token_metadata(tokenizer) | dict(grounding_candidates=GROUNDING_CANDIDATES,
        input_protocol=INPUT_PROTOCOL, candidate_binding='native temporal group index; unrounded score_times')


def _replacement(processor, record, *, fast=True):
    grid = record['grid_thw']
    if (processor.video_processor.temporal_patch_size != 2 or len(grid) != 3
            or grid[0] != (record['frames']+1)//2):
        raise ValueError('Timestamp options require the native two-frame temporal grouping')
    fps = float(record.get('effective_fps', record['fps']))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Timestamp metadata requires a positive effective sampling fps')
    metadata = VideoMetadata(total_num_frames=record['frames'], fps=fps,
        frames_indices=list(range(record['frames'])))
    shape = [grid[0], 2, 2] if fast else grid
    return processor.replace_video_token(dict(video_grid_thw=np.asarray([shape]),
        video_metadata=[metadata]), video_idx=0)


def timestamp_texts(processor, record, *, replacement=None):
    """Exact official strings in temporal-group order, from metadata only."""
    replacement = _replacement(processor, record) if replacement is None else replacement
    texts = _TIMESTAMP.findall(replacement)
    raw = processor._calculate_timestamps(list(range(record['frames'])),
        record.get('effective_fps', record['fps']), processor.video_processor.temporal_patch_size)
    scores = record.get('score_times')
    if (scores is None or len(texts) != record['grid_thw'][0] or len(raw) != len(texts)
            or len(scores) != len(texts)
            or not np.allclose(raw, scores, rtol=1e-10, atol=1e-8)
            or texts != [f'<{value:.1f} seconds>' for value in raw]):
        raise ValueError('Official displayed timestamps must map one-to-one to native score midpoints')
    return texts


def suffix(row, *, timestamp_texts=None):
    task = validate_row(row)
    original = original_suffix(row)
    if task == CHOICE:
        return original
    if (not timestamp_texts or not all(isinstance(text, str) and _TIMESTAMP.fullmatch(text)
                                     for text in timestamp_texts)):
        raise ValueError('Grounding requires its complete ordered native timestamp option list')
    return original[:-len(DECIDE)] + 'Options:\n' + ''.join(
        text + OPTION_END + '\n' for text in timestamp_texts) + DECIDE


def prompt(processor, row, *, timestamp_texts=None):
    content = [dict(type='video', video='provided_in_memory'),
        dict(type='text', text=suffix(row, timestamp_texts=timestamp_texts))]
    return processor.apply_chat_template([dict(role='user', content=content)],
        tokenize=False, add_generation_prompt=False)


def cached_inputs(processor, record, row, *, fast=True, lengths_only=False):
    replacement = _replacement(processor, record, fast=fast)
    options = timestamp_texts(processor, record, replacement=replacement) if task_of(row) == GROUNDING else None
    texts, _ = processor.get_text_with_replacements(
        [prompt(processor, row, timestamp_texts=options)], [], [replacement], [])
    native = processor(text=texts, return_tensors=None)
    native.convert_to_tensors('pt')
    inputs = {key: native[key] for key in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
    grid = record['grid_thw']
    slots = inputs['input_ids'][0] == processor.video_token_id
    spatial = grid[1]*grid[2]//4
    if record['tokens'] != grid[0]*spatial or int(slots.sum()) != (grid[0] if fast else record['tokens']):
        raise ValueError('Timestamp options must preserve every original native video token')
    length = inputs['input_ids'].shape[1] + (grid[0]*(spatial-1) if fast else 0)
    if lengths_only:
        return length
    if fast:
        repeat = torch.ones(inputs['input_ids'].shape[1], dtype=torch.long)
        repeat[slots] = spatial
        inputs = {key: torch.repeat_interleave(value, repeat, dim=1) for key, value in inputs.items()}
    if inputs['input_ids'].shape[1] != length:
        raise ValueError('Exact timestamp-option token expansion changed length')
    return inputs | {'video_grid_thw': torch.tensor([grid], dtype=torch.long)}


def readout_indices(tokenizer, ids, task):
    if task not in (GROUNDING, CHOICE):
        raise ValueError('Unknown timestamp-option decision task')
    decide = (ids == tokenizer.convert_tokens_to_ids(DECIDE)).nonzero().flatten()
    candidates = (ids == tokenizer.convert_tokens_to_ids(OPTION_END)).nonzero().flatten()
    if len(decide) != 1 or not len(candidates) or not (candidates < decide[0]).all():
        raise ValueError('Every option must end before exactly one DECIDE marker')
    return candidates, decide[0]
