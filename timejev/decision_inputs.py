"""Versioned typed-decision inputs; the native video token sequence is unchanged."""
import math
import torch
from transformers.video_utils import VideoMetadata
from .vision import load_processor as native_processor

FORMULATION = 'timejev_decision_v1'
DECIDE = '<|decide|>'
OPTION_END = '<|option_end|>'
SPECIAL_TOKENS = [DECIDE, OPTION_END]
GROUNDING, CHOICE = 0, 1


def load_processor(base):
    processor = native_processor(base)
    processor.tokenizer.add_special_tokens({'extra_special_tokens': SPECIAL_TOKENS},
                                            replace_extra_special_tokens=False)
    for token in SPECIAL_TOKENS:
        ids = processor.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] != processor.tokenizer.convert_tokens_to_ids(token):
            raise ValueError(f'Decision marker is not a single token: {token}')
    return processor


def token_metadata(tokenizer):
    return dict(formulation=FORMULATION, tokens={t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS},
                tokenizer_size=len(tokenizer), initialization='copy native im_end embedding',
                query_placement='last', attention_mode='causal', shared_head=True)


def task_of(row):
    task = row.get('task', 'grounding')
    if task not in ('grounding', 'choice'):
        raise ValueError(f'Unknown decision task: {task}')
    return GROUNDING if task == 'grounding' else CHOICE


def validate_row(row, *, require_labels=False):
    task = task_of(row)
    texts = [row['query']]
    if task == CHOICE:
        options = row.get('options')
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError('Choice requires at least two option strings')
        texts += options
        answer = row.get('answer_index')
        if (require_labels or answer is not None) and (type(answer) is not int or not 0 <= answer < len(options)):
            raise ValueError('answer_index must index an option (zero-based)')
    elif require_labels and not row.get('spans'):
        raise ValueError('Grounding training requires one or more GT spans')
    for text in texts:
        if not isinstance(text, str) or not text.strip() or '<|' in text:
            raise ValueError('Query/options must be nonempty text without control-token syntax')
    for start, end in row.get('spans', []) if task == GROUNDING else []:
        # Keep the existing data protocol: annotation endpoints may be rounded
        # beyond media duration. Preserve GT and label the observed midpoints;
        # do not silently clip annotations or drop these training samples.
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError('Invalid grounding span')
    return task


def suffix(row):
    task = validate_row(row)
    instruction = 'temporal grounding' if task == GROUNDING else 'single-choice video question answering'
    text = f'\nTask: {instruction}\nQuestion: {row["query"]}\n'
    if task == CHOICE:
        text += 'Options:\n' + ''.join(option + OPTION_END + '\n' for option in row['options'])
    return text + DECIDE


def prompt(processor, row):
    content = [dict(type='video', video='provided_in_memory'), dict(type='text', text=suffix(row))]
    return processor.apply_chat_template([dict(role='user', content=content)],
                                         tokenize=False, add_generation_prompt=False)


def cached_inputs(processor, record, row, *, fast=True, lengths_only=False):
    """Official timestamps and tokenizer; optional exact spatial-token expansion."""
    grid = record['grid_thw']
    shape = [grid[0], 2, 2] if fast else grid
    metadata = VideoMetadata(total_num_frames=record['frames'],
        fps=record.get('effective_fps', record['fps']), frames_indices=list(range(record['frames'])))
    replacement = processor.replace_video_token(
        dict(video_grid_thw=torch.tensor([shape]), video_metadata=[metadata]), video_idx=0)
    texts, _ = processor.get_text_with_replacements([prompt(processor, row)], [], [replacement], [])
    native = processor(text=texts, return_tensors=None)
    native.convert_to_tensors('pt')
    inputs = {k: native[k] for k in ['input_ids', 'attention_mask', 'mm_token_type_ids']}
    slots = inputs['input_ids'][0] == processor.video_token_id
    spatial = grid[1] * grid[2] // 4
    assert record['tokens'] == grid[0] * spatial
    assert int(slots.sum()) == (grid[0] if fast else record['tokens'])
    n = inputs['input_ids'].shape[1] + (grid[0] * (spatial - 1) if fast else 0)
    if lengths_only:
        return n
    if fast:
        repeat = torch.ones(inputs['input_ids'].shape[1], dtype=torch.long)
        repeat[slots] = spatial
        inputs = {k: torch.repeat_interleave(v, repeat, dim=1) for k, v in inputs.items()}
    assert inputs['input_ids'].shape[1] == n
    return inputs | {'video_grid_thw': torch.tensor([grid], dtype=torch.long)}


def readout_indices(tokenizer, ids, task):
    decide = (ids == tokenizer.convert_tokens_to_ids(DECIDE)).nonzero().flatten()
    if len(decide) != 1:
        raise ValueError('Every sample must contain exactly one DECIDE token')
    if task == CHOICE:
        candidates = (ids == tokenizer.convert_tokens_to_ids(OPTION_END)).nonzero().flatten()
    else:
        candidates = ((ids == tokenizer.convert_tokens_to_ids('<|vision_end|>')) &
            (torch.roll(ids, 1) == tokenizer.convert_tokens_to_ids('<|video_pad|>'))).nonzero().flatten()
    if not len(candidates) or not (candidates < decide[0]).all():
        raise ValueError('Decision candidates must precede DECIDE')
    return candidates, decide[0]
