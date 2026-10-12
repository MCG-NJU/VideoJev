"""Exact lengths from persistent video metadata and short prompt tokenization.

No feature tensor, image or video is opened here. Native video expansion is
performed once per video geometry/tokenizer version, never once per query.
The first timestamp stays with the prompt prefix: an arbitrary BPE split there
would otherwise make adding separately tokenized lengths unsafe.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from transformers.video_utils import VideoMetadata

from .engine import save_json

SCHEMA = 'qwen35_exact_token_metadata_v1'


def digest(value):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def tokenizer_identity(processor, *, prompt_version):
    tokenizer = processor.tokenizer
    if tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError('Metadata lengths require a tokenizer without implicit BOS/EOS additions')
    for token in (processor.vision_start_token, processor.vision_end_token, processor.video_token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        added = tokenizer.added_tokens_decoder.get(token_id)
        if (tokenizer.encode(token, add_special_tokens=False) != [token_id]
                or added is None or not added.special or added.lstrip or added.rstrip):
            raise ValueError(f'Unsafe metadata token boundary: {token}')
    return dict(schema=SCHEMA, tokenizer_sha256=digest(tokenizer.backend_tokenizer.to_str()),
        chat_template_sha256=digest(tokenizer.chat_template), prompt_version=prompt_version,
        video_expansion_sha256=digest(inspect.getsource(processor.replace_video_token)),
        timestamp_function_sha256=digest(inspect.getsource(processor._calculate_timestamps)),
        merge_size=processor.video_processor.merge_size,
        temporal_patch_size=processor.video_processor.temporal_patch_size)


def record_identity(record):
    """Only fields consumed by native video text expansion; no media stat/read."""
    grid = [int(v) for v in record['grid_thw']]
    frames = int(record['frames'])
    if (len(grid) != 3 or min(grid) <= 0 or grid[1] % 2 or grid[2] % 2
            or grid[0] != (frames + 1) // 2 or record['tokens'] != int(np.prod(grid)) // 4):
        raise ValueError('Invalid native video grid/token/frame metadata')
    effective_fps = float(record.get('effective_fps', record['fps']))
    if not np.isfinite(effective_fps) or effective_fps <= 0:
        raise ValueError('Invalid metadata effective fps')
    return dict(cache_id=record['cache_id'], grid_thw=grid, frames=frames,
                tokens=int(record['tokens']), fps=float(record['fps']), effective_fps=effective_fps)


def _video_entry(processor, record):
    identity = record_identity(record)
    grid = identity['grid_thw']
    metadata = VideoMetadata(total_num_frames=identity['frames'], fps=identity['effective_fps'],
                             frames_indices=list(range(identity['frames'])))
    # A NumPy grid carries shape metadata; no visual tensors are instantiated.
    replacement = processor.replace_video_token(dict(video_grid_thw=np.array([[grid[0], 2, 2]]),
                                                       video_metadata=[metadata]), video_idx=0)
    head, marker, tail = replacement.partition(processor.vision_start_token)
    body = marker + tail
    if not marker or not body.endswith(processor.vision_end_token):
        raise ValueError('Native video expansion no longer has special-token boundaries')
    ids = processor.tokenizer.encode(body, add_special_tokens=False)
    if ids.count(processor.video_token_id) != grid[0]:
        raise ValueError('Native skeleton video token count changed')
    return dict(identity=identity, prefix_timestamp=head,
                body_tokens=len(ids) + identity['tokens'] - grid[0])


def prepare_video_metadata(processor, records, path, *, prompt_version, workers=8):
    """Load/reuse a compact JSON catalogue, rebuilding only invalidated entries."""
    began = time.monotonic()
    path = Path(path)
    identity = tokenizer_identity(processor, prompt_version=prompt_version)
    old = json.loads(path.read_text()) if path.exists() else {}
    reusable = old.get('entries', {}) if old.get('identity') == identity else {}
    entries, missing = {}, []
    for key, record in records.items():
        entry = reusable.get(key)
        if entry is not None and entry['identity'] == record_identity(record):
            entries[key] = entry
        else:
            missing.append(key)
    def build(key):
        return key, _video_entry(processor, records[key])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries.update(pool.map(build, missing))
    # Retain entries for other data subsets only when tokenizer/prompt identity
    # still matches. A later 10%/30%/100% run can reuse this same catalogue.
    if missing or not path.exists() or old.get('identity') != identity:
        save_json(dict(identity=identity, entries=reusable | entries), path)
    return entries, dict(path=str(path.resolve()), identity=identity, videos=len(records),
        reused_videos=len(records)-len(missing), built_videos=len(missing),
        elapsed_seconds=time.monotonic()-began)


def exact_lengths(processor, rows, video_entries, prompt_fn, *, batch_size=256, progress=None):
    """Tokenize only short query/option text in batches; never call processor().

    Both cuts are adjacent to native registered special tokens, which cannot
    participate in ordinary-text BPE merges. The original prompt, including its
    outer vision wrappers and first timestamp, is preserved exactly.
    """
    entries = {}
    began = time.monotonic()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        fragments, video_counts = [], []
        for row in batch:
            video = video_entries[row['video_key']]
            text = prompt_fn(processor, row)
            if text.count(processor.video_token) != 1:
                raise ValueError('Expected exactly one native video placeholder per sample')
            prefix, suffix = text.split(processor.video_token)
            fragments.extend((prefix + video['prefix_timestamp'], suffix))
            video_counts.append(video['body_tokens'])
        encoded = processor.tokenizer(fragments, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False)['input_ids']
        for i, row in enumerate(batch):
            if row['id'] in entries:
                raise ValueError(f'Duplicate sample ID: {row["id"]}')
            entries[row['id']] = dict(tokens=len(encoded[2*i])+video_counts[i]+len(encoded[2*i+1]),
                visual_tokens=video_entries[row['video_key']]['identity']['tokens'])
        if progress:
            seconds = time.monotonic()-began
            progress(dict(stage='exact_token_lengths_from_metadata',completed=len(entries),total=len(rows),
                elapsed_seconds=seconds,queries_per_second=len(entries)/max(seconds,1e-9),
                remaining_seconds=(len(rows)-len(entries))*seconds/len(entries)))
    return entries
