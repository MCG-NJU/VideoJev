"""Convert a completed VideoJev task checkpoint into a standalone HF model directory."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from timejev.hf_checkpoint import export_tensors, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expected-id', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--decoder-threshold', type=float, default=.25, help='Grounding threshold (default: 0.25)')
    args = parser.parse_args()
    if not 0 < args.decoder_threshold < 1:
        parser.error("--decoder-threshold must be between 0 and 1")
    path, base = Path(args.checkpoint), Path(args.base)
    native = json.loads((base/'config.json').read_text())
    if path.is_file():
        source_hash = file_sha256(path)
        if args.expected_sha256 and source_hash != args.expected_sha256:
            raise ValueError('Source checkpoint SHA256 differs from the final record')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if checkpoint['checkpoint_id'] != args.expected_id or checkpoint['config']['training_stage'] != 'joint':
            raise ValueError('Expected the verified final joint checkpoint')
        tensors = checkpoint['model'].items()
    else:
        metadata = json.loads((path/'task_model_config.json').read_text())
        completion = json.loads((path/'complete_manifest.json').read_text())
        if (metadata['checkpoint_id'] != args.expected_id or not completion['complete']
                or completion['head_only'] or metadata['recipe']['training_stage'] != 'joint'):
            raise ValueError('Expected the verified complete 35B joint export')
        if file_sha256(base/'config.json') != metadata['vision_reference']['config_sha256']:
            raise ValueError('Frozen vision base configuration differs from the source export')
        hashes = {name: file_sha256(path/name) for name in completion['files']}
        hashes['complete_manifest.json'] = file_sha256(path/'complete_manifest.json')
        import hashlib
        source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if args.expected_sha256 and source_hash != args.expected_sha256:
            raise ValueError('Source export SHA256 differs from the final record')
        weight_map = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        vision_map = json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
        def stream():
            for filename in sorted(set(weight_map.values())):
                with safe_open(str(path/filename), framework='pt') as source:
                    for name in source.keys():
                        if name.startswith('language_model.'):
                            target = 'backbone.'+name
                        elif name.startswith('head.'):
                            target = name
                        else:
                            raise ValueError(f'Unknown task parameter: {name}')
                        yield target, source.get_tensor(name)
            for filename in sorted({f for k, f in vision_map.items() if k.startswith('model.visual.')}):
                with safe_open(str(base/filename), framework='pt') as source:
                    for name in source.keys():
                        if name.startswith('model.visual.'):
                            yield 'backbone.'+name.removeprefix('model.'), source.get_tensor(name)
        tensors = stream()
    result = export_tensors(tensors, args.out, base=base, native_config=native,
        checkpoint_id=args.expected_id,
        decoder=dict(threshold=args.decoder_threshold, smooth=1, postprocess='round_repair_clamp_v1'),
        metadata=dict(source_sha256=source_hash, precision_policy='match_input_model_weight_dtype',
                      frozen_vision_included=True))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
