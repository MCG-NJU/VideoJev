"""RGB-only MCQ evaluation contracts at the fixed 256K context."""
import hashlib
import json
from pathlib import Path

from .rgb_data import geometries, text_item
from .rgb_media import metadata, fingerprint

from .qa_eval_settings import QA_CONTEXT

CONTEXTS = (QA_CONTEXT,)
PROTOCOL = 'qwen35_rgb_mcq_context80_v1'
COUNTS = dict(lvbench=1549, videomme=2700, longvideobench=1337, mlvu=2174)
VIDEO_COUNTS = dict(lvbench=103, videomme=900, longvideobench=753, mlvu=1122)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def visual_budget(context):
    if context not in CONTEXTS:
        raise ValueError(f'Unsupported context: {context}')
    return context * 4 // 5


def full_frame_record(row, name):
    record = metadata(row, name)
    state = json.loads((Path(record['source']['path'])/'.complete.json').read_text())
    # The requested input is the complete extracted frame sequence. Annotation
    # durations rounded to seconds must not discard its final JPEG.
    record.update(frames=int(state['frame_count']), sampled_duration=float(state['media_duration']),
                  media_duration=float(state['media_duration']))
    if not 0 < record['frames'] <= 2048 or record['fps'] != 2.:
        raise ValueError('Expected the complete 2fps / 2048-frame extraction')
    return record


def check_source(record):
    root = Path(record['source']['path'])
    for filename, key in (('.complete.json', 'identity'), ('.timestamps.jsonl', 'timestamps_identity')):
        if fingerprint(root/filename) != record['source'][key]:
            raise ValueError(f'RGB extraction changed since preparation: {root/filename}')


def choose_geometry(record, nonvisual_tokens, context):
    groups = (record['frames']+1)//2
    for h, w in geometries(*record['original_hw'], minimum=1024):
        tokens = groups*(h//32)*(w//32)
        if tokens <= visual_budget(context) and tokens+nonvisual_tokens <= context:
            return dict(processed_hw=[h, w], grid_thw=[groups, h//16, w//16],
                        tokens=tokens, pixels_per_frame=h*w)
    raise ValueError('Complete question cannot fit even at the smallest native spatial grid')


def question_item(processor, row, video, question, context):
    item = text_item(processor, [row], video['record'],
                     dict(id=row['id'], geometry=video['geometry'], tokens=question['tokens']))
    item['audit'].update(rgb_protocol=PROTOCOL, feature_cache_id=PROTOCOL,
        max_sequence_length=context, visual_token_budget=visual_budget(context),
        visual_budget_fraction=.8, total_input_tokens=question['tokens'])
    return item


def validate_spec(spec, rows, context):
    seen = []
    for video in spec['videos']:
        record, shape = video['record'], video['geometry']
        h, w = shape['processed_hw']
        groups = (record['frames']+1)//2
        if (not 0 < record['frames'] <= 2048 or record['fps'] != 2.
                or record['source']['kind'] != 'frames_sidecar'
                or h % 32 or w % 32 or h*w != shape['pixels_per_frame']
                or not 1024 <= h*w <= 640*640
                or shape['grid_thw'] != [groups, h//16, w//16]
                or shape['tokens'] != groups*h*w//1024
                or shape['tokens'] > visual_budget(context)):
            raise ValueError('Invalid RGB geometry or visual budget')
        for q in video['questions']:
            row = rows[q['query_index']]
            if (row['id'] != q['id'] or row['video_key'] != video['video_key']
                    or not shape['tokens'] < q['tokens'] <= context
                    or row['task'] != 'choice'):
                raise ValueError('Question identity/context differs from RGB plan')
            seen.append(q['query_index'])
    if sorted(seen) != list(range(len(rows))):
        raise ValueError('RGB evaluation must cover each question exactly once')


def assignments(spec, world):
    """Deterministic longest-first allocation; keep all questions of a video together."""
    loads, shards = [0]*world, [[] for _ in range(world)]
    def cost(i):
        v = spec['videos'][i]
        n = max(q['tokens'] for q in v['questions'])
        return n*n + n*2048*len(v['questions'])
    for i in sorted(range(len(spec['videos'])), key=lambda i: (-cost(i), spec['videos'][i]['video_key'])):
        rank = min(range(world), key=lambda r: (loads[r], r))
        shards[rank].append(i)
        loads[rank] += cost(i)
    return shards
