"""Canonical VideoJev prompts, intact query groups, and video preprocessing."""
import math
import re

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from .videojev_temporal import pair_timeline, decode
from .videojev_video import read_video

DECIDE = '<|decide|>'
OPTION_END = '<|option_end|>'


def suffix(row, stamps):
    task = row.get('task', 'grounding')
    if task not in ('grounding', 'choice'):
        raise ValueError('task must be grounding or choice')
    for text in [row['query'], *row.get('options', [])]:
        if not isinstance(text, str) or not text.strip() or '<|' in text:
            raise ValueError('Questions/options must be nonempty text without control tokens')
    if task == 'choice' and len(row.get('options', [])) < 2:
        raise ValueError('Choice questions need at least two options')
    instruction = 'temporal grounding' if task == 'grounding' else 'single-choice video question answering'
    text = f'\nTask: {instruction}\nQuestion: {row["query"]}\nOptions:\n'
    candidates = stamps if task == 'grounding' else row['options']
    return text + ''.join(value + OPTION_END + '\n' for value in candidates) + DECIDE


def native_inputs(processor, record, rows):
    grid = record['grid_thw']
    meta = VideoMetadata(total_num_frames=record['frames'],
        fps=record.get('effective_fps', record['fps']), frames_indices=list(range(record['frames'])))
    replacement = processor.replace_video_token(dict(video_grid_thw=np.asarray([[grid[0], 2, 2]]),
        video_metadata=[meta]), video_idx=0)
    stamps = re.findall(r'<[0-9]+\.[0-9] seconds>', replacement)
    raw_times = processor._calculate_timestamps(list(range(record['frames'])), meta.fps, 2)
    if len(stamps) != grid[0] or not np.allclose(raw_times, record['score_times'], rtol=1e-10, atol=1e-8):
        raise ValueError('Timestamp candidates do not match native video midpoints')
    texts = []
    for i, row in enumerate(rows):
        content = ([dict(type='video', video='provided_in_memory')] if i == 0 else [])
        content += [dict(type='text', text=suffix(row, stamps))]
        text = processor.apply_chat_template([dict(role='user', content=content)],
            tokenize=False, add_generation_prompt=False)
        if i == 0:
            replaced, _ = processor.get_text_with_replacements([text], [], [replacement], [])
            value = processor(text=replaced, return_tensors='pt')
            spatial = grid[1] * grid[2] // 4
            slots = value['input_ids'][0] == processor.video_token_id
            if int(slots.sum()) != grid[0] or record['tokens'] != grid[0]*spatial:
                raise ValueError('Native grid/token count mismatch')
            repeats = torch.ones(value['input_ids'].shape[1], dtype=torch.long)
            repeats[slots] = spatial
            first_ids = torch.repeat_interleave(value['input_ids'], repeats, dim=1)
            first_types = torch.repeat_interleave(value['mm_token_type_ids'], repeats, dim=1)
        else:
            texts.append(torch.tensor([processor.tokenizer.encode(text, add_special_tokens=False)]))
    return torch.cat([first_ids, *texts], dim=1), torch.cat([first_types, *[torch.zeros_like(t) for t in texts]], dim=1)


def prepare_query_groups(processor, record, questions, context):
    if not questions or context < 64:
        raise ValueError('Provide questions and a context of at least 64 tokens')
    if len({q.get('id', i) for i, q in enumerate(questions)}) != len(questions):
        raise ValueError('Question IDs must be unique')
    # Query lengths are computed with the exact native tokenizer. No labels enter
    # this function, and no answer history is carried from one group to the next.
    start = 0
    decide_id = processor.tokenizer.convert_tokens_to_ids(DECIDE)
    option_id = processor.tokenizer.convert_tokens_to_ids(OPTION_END)
    while start < len(questions):
        end = start + 1
        ids, types = native_inputs(processor, record, questions[start:end])
        if (ids.shape[1]+63)//64*64 > context:
            raise ValueError(f'A complete query exceeds context; no question was skipped: {start}')
        while end < len(questions):
            candidate_ids, candidate_types = native_inputs(processor, record, questions[start:end+1])
            if (candidate_ids.shape[1]+63)//64*64 > context:
                break
            ids, types = candidate_ids, candidate_types
            end += 1
        rows = questions[start:end]
        query_indices = (ids[0] == decide_id).nonzero().flatten()
        option_indices = (ids[0] == option_id).nonzero().flatten()
        counts = [len(row['options']) if row.get('task') == 'choice' else record['grid_thw'][0] for row in rows]
        if len(query_indices) != len(rows) or len(option_indices) != sum(counts):
            raise ValueError('Incomplete decision/candidate markers')
        pairs = torch.zeros((len(rows), max(counts)), dtype=torch.long)
        mask = torch.zeros_like(pairs, dtype=torch.bool)
        offset, previous = 0, -1
        for i, count in enumerate(counts):
            selected = option_indices[offset:offset+count]
            if not ((selected > previous) & (selected < query_indices[i])).all():
                raise ValueError('Candidate readouts cross query boundaries')
            pairs[i, :count], mask[i, :count] = selected, True
            previous = query_indices[i]
            offset += count
        yield dict(input_ids=ids, mm_token_type_ids=types,
            video_grid_thw=torch.tensor([record['grid_thw']]), pair_indices=pairs,
            query_indices=query_indices, score_mask=mask,
            decision_tasks=torch.tensor([int(row.get('task') == 'choice') for row in rows])), rows
        start = end


def prepare_video(processor, path, *, fps=2., max_frames=2048, context=196608):
    import av
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else container.duration/1e6
    sample = read_video(dict(video_path=str(path), duration=duration), fps, max_frames=max_frames)
    n, height, width, _ = sample['video'].shape
    vp = processor.video_processor
    budget = context * 4 // 5
    def geometry(cap):
        h, w = smart_resize(n, height, width, temporal_factor=vp.temporal_patch_size,
            factor=vp.patch_size*vp.merge_size, min_pixels=vp.size['shortest_edge'], max_pixels=n*cap)
        return h, w, (n+1)//2*(h//32)*(w//32)
    low, high = max(1024, math.ceil(vp.size['shortest_edge']/n)), 640*640
    if geometry(low)[2] > budget:
        raise ValueError('Complete video cannot fit the context at minimum native resolution')
    while low < high:
        mid = (low+high+1)//2
        if geometry(mid)[2] <= budget:
            low = mid
        else:
            high = mid-1
    meta = VideoMetadata(total_num_frames=n, fps=sample['fps'], frames_indices=list(range(n)))
    native = vp(videos=[sample['video']], video_metadata=[meta], do_sample_frames=False,
        size=dict(shortest_edge=vp.size['shortest_edge'], longest_edge=n*low), return_tensors='pt')
    grid = native['video_grid_thw'][0].tolist()
    times, bounds = pair_timeline(sample['times'], sample['sampled_duration'])
    record = dict(frames=n, fps=sample['fps'], grid_thw=grid, tokens=int(np.prod(grid))//4,
        score_times=times.tolist(), score_bounds=bounds.tolist(), sampled_duration=sample['sampled_duration'])
    return native['pixel_values_videos'], record


def format_predictions(output, rows, record, decoder):
    results = []
    for i, row in enumerate(rows):
        count = int(output.score_mask[i].sum())
        logits = output.logits[i, :count].float().cpu().tolist()
        probabilities = output.probabilities[i, :count].float().cpu().tolist()
        value = dict(id=row.get('id', i), task=row.get('task', 'grounding'),
                     probabilities=probabilities)
        if row.get('task') == 'choice':
            answer = int(np.argmax(probabilities))
            value.update(answer_index=answer, answer=row['options'][answer])
        else:
            value['interval'] = decode(logits, record['score_times'], record['sampled_duration'],
                decoder['threshold'], decoder['smooth'], bounds=record['score_bounds'],
                postprocess=decoder['postprocess'])
        results.append(value)
    return results
