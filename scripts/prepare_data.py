"""Prepare custom JSONL data and ready-to-train RGB configs for either model."""
import argparse
import json
from pathlib import Path

from timejev.config import read_config
from timejev.engine import save_json, sha256
from timejev.user_data import load_manifest, split_stages, write_jsonl


def prepare(manifest, out, base, model_size='4B', workers=8, head_fraction=.3, world_size=64):
    if world_size < 1 or 64 % world_size:
        raise ValueError('world_size must divide the global batch size of 64')
    if workers < 1:
        raise ValueError('workers must be positive')
    base, out = Path(base).resolve(), Path(out).resolve()
    native = json.loads((base/'config.json').read_text())
    expected = 'qwen3_5' if model_size == '4B' else 'qwen3_5_moe'
    if native['model_type'] != expected:
        raise ValueError(f'{model_size} requires a {expected} base model')
    rows = load_manifest(manifest)
    stages = split_stages(rows, head_fraction)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    source = out/'source'
    for stage, members in stages.items():
        write_jsonl(members, source/f'data/{stage}/train.jsonl')
    save_json(dict(seed=20260924, head_fraction=head_fraction, source_sha256=sha256(manifest),
        stages={s: dict(questions=len(v), videos=len({r['video_key'] for r in v})) for s, v in stages.items()}),
        source/'data/split.json')
    if model_size == '4B':
        from scripts.prepare_rgb192 import prepare as prepare_rgb
        prefix, evaluation = 'rgb192', 'evaluation.json'
    else:
        from scripts.prepare_qwen36_mixed import prepare as prepare_rgb
        prefix, evaluation = 'qwen36', 'evaluation_147456.json'
    prepare_rgb(source, out, str(base), workers, benchmarks=())
    ready = json.loads((out/'ready.json').read_text())
    if any(s.get('dropped_questions', 0) for s in ready['summary'].values()):
        (out/'ready.json').unlink()
        raise ValueError('Some questions exceed the context limit; inspect filter_audit.json and shorten them')
    root = Path(__file__).resolve().parents[1]
    for stage in stages:
        config = read_config(root/f'configs/{prefix}_{stage}.json')
        config.update(base=str(base), train=str(out/stage/'train.jsonl'), benchmarks={},
            packed_train_lengths=str(out/stage/'episodes.json'), packed_train_plan=str(out/stage/'packs.json'),
            packed_eval_lengths={}, training_split_report=str(out/'split.json'),
            rgb_metadata=str(out/'rgb_metadata.json'), rgb_evaluation=str(out/evaluation),
            rgb_ready=str(out/'ready.json'), workers=workers, expected_world_size=world_size,
            accumulation=64//world_size, training_fraction=head_fraction if stage == 'head_only' else 1-head_fraction)
        config.pop('qa_option_permutation', None)
        ds = read_config(config['deepspeed'])
        ds.update(train_batch_size=64, gradient_accumulation_steps=64//world_size)
        save_json(ds, out/'configs/deepspeed.json')
        config['deepspeed'] = str(out/'configs/deepspeed.json')
        save_json(config, out/f'configs/{stage}.json')
    ready['files'].update({str(p.relative_to(out)): sha256(p) for p in (out/'configs').glob('*.json')})
    save_json(ready, out/'ready.json')
    return ready['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', required=True, help='Local Qwen base model directory')
    parser.add_argument('--model-size', choices=('4B', '35B-A3B'), default='4B')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head-fraction', type=float, default=.3)
    parser.add_argument('--world-size', type=int, default=64, help='Total training GPUs; must divide 64')
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
