"""Bounded-memory HF exports, separate from full-precision training recovery."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def checkpoint_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    inventory = {p.name: file_sha256(p) for p in sorted(path.iterdir())
                 if p.is_file() and p.name != 'export_complete.json'}
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def checkpoint_stat(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def base_weight_dtype(base):
    """Read actual input weight precision instead of the optimizer/master dtype."""
    base = Path(base)
    index = base/'model.safetensors.index.json'
    if index.exists():
        weights = json.loads(index.read_text())['weight_map']
        key = next(k for k in weights if k.endswith('embed_tokens.weight'))
        with safe_open(str(base/weights[key]), framework='pt') as source:
            dtype = source.get_slice(key).get_dtype()
    else:
        with safe_open(str(base/'model.safetensors'), framework='pt') as source:
            key = next(k for k in source.keys() if k.endswith('embed_tokens.weight'))
            dtype = source.get_slice(key).get_dtype()
    choices = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
    if dtype not in choices:
        raise ValueError(f'Unsupported base weight dtype: {dtype}; explicit dequantization is required')
    return choices[dtype]


def _clean_native(value):
    if isinstance(value, dict):
        return {k: _clean_native(v) for k, v in value.items() if k not in ('_name_or_path', '_commit_hash')}
    if isinstance(value, list):
        return [_clean_native(v) for v in value]
    return value


def export_tensors(tensors, directory, *, base, native_config, checkpoint_id,
                   decoder=None, metadata=None, max_shard_bytes=5_000_000_000,
                   training_state=None):
    """Cast to the input model dtype and stream verified, standard HF shards."""
    from transformers import AutoProcessor
    from videojev_hf.configuration_videojev import VideoJevConfig
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f'Choose a new export directory: {directory}')
    if max_shard_bytes <= 0:
        raise ValueError('max_shard_bytes must be positive')
    staging = directory.with_name(directory.name+'.incomplete')
    if staging.exists():
        raise FileExistsError(f'Inspect the incomplete export before retrying: {staging}')
    staging.mkdir(parents=True)
    dtype = base_weight_dtype(base)
    processor = AutoProcessor.from_pretrained(base)
    processor.tokenizer.add_special_tokens({'extra_special_tokens': ['<|decide|>', '<|option_end|>']},
                                           replace_extra_special_tokens=False)
    processor.save_pretrained(staging)
    native = _clean_native(native_config)
    native['dtype'] = str(dtype).removeprefix('torch.')
    for key in ('text_config', 'vision_config'):
        native[key]['dtype'] = native['dtype']
    config = VideoJevConfig(backbone_config=native, dtype=native['dtype'], checkpoint_id=checkpoint_id,
        decide_token_id=processor.tokenizer.convert_tokens_to_ids('<|decide|>'),
        option_end_token_id=processor.tokenizer.convert_tokens_to_ids('<|option_end|>'), decoder=decoder)
    config.save_pretrained(staging)
    runtime = Path(__file__).resolve().parents[1]/'videojev_hf'
    for path in runtime.glob('*.py'):
        if path.name != '__init__.py':
            shutil.copy2(path, staging/path.name)
    pending, pending_bytes, total, count, shards, mapping = {}, 0, 0, 0, [], {}
    dtype_counts = {}

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f'model-{len(shards)+1:05d}.safetensors'
        save_file(pending, staging/filename, metadata={'format': 'pt'})
        # Verify serialized values against the expected cast before releasing RAM.
        with safe_open(str(staging/filename), framework='pt') as saved:
            if set(saved.keys()) != set(pending):
                raise ValueError('Export tensor inventory changed')
            for key, value in pending.items():
                if not torch.equal(saved.get_tensor(key), value):
                    raise ValueError(f'Export round-trip mismatch: {key}')
        for key in pending:
            mapping[key] = filename
        shards.append(filename)
        print(json.dumps(dict(event='hf_shard_saved', shard=len(shards), bytes=pending_bytes)), flush=True)
        pending, pending_bytes = {}, 0

    for name, tensor in tensors:
        name = name.replace('_checkpoint_wrapped_module.', '').replace('_orig_mod.', '')
        if name in mapping or name in pending:
            raise ValueError(f'Duplicate exported key: {name}')
        value = tensor.detach().to(device='cpu', dtype=dtype if tensor.is_floating_point() else tensor.dtype).contiguous()
        size = value.numel()*value.element_size()
        if pending and pending_bytes+size > max_shard_bytes:
            flush()
        pending[name] = value
        pending_bytes += size
        total += size
        count += 1
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0)+value.numel()
    flush()
    if not count:
        raise ValueError('Refuse to export an empty model')
    for index, old in enumerate(shards, 1):
        new = f'model-{index:05d}-of-{len(shards):05d}.safetensors'
        (staging/old).rename(staging/new)
        for key in mapping:
            if mapping[key] == old:
                mapping[key] = new
    (staging/'model.safetensors.index.json').write_text(json.dumps(
        dict(metadata=dict(total_size=total), weight_map=mapping), indent=2)+'\n')
    if training_state is not None:
        (staging/'training_state.json').write_text(json.dumps(training_state, indent=2)+'\n')
    info = dict(checkpoint_id=checkpoint_id, dtype=str(dtype), tensors=count, tensor_bytes=total,
                shards=len(shards), verified_roundtrip=True, **(metadata or {}))
    (staging/'conversion.json').write_text(json.dumps(info, indent=2)+'\n')
    files = {p.name: dict(bytes=p.stat().st_size, sha256=file_sha256(p)) for p in staging.iterdir() if p.is_file()}
    (staging/'export_complete.json').write_text(json.dumps(dict(complete=True, checkpoint_id=checkpoint_id,
        files=files, dtype_counts=dtype_counts), indent=2)+'\n')
    staging.rename(directory)
    return info


def export_training_checkpoint(checkpoint, directory):
    """Called on rank zero after model gathering; never serialize optimizer state here."""
    config = checkpoint['config']
    native = json.loads((Path(config['base'])/'config.json').read_text())
    state = {key: value for key, value in checkpoint.items() if key != 'model'}
    return export_tensors(checkpoint['model'].items(), directory, base=config['base'],
        native_config=native, checkpoint_id=checkpoint['checkpoint_id'], training_state=state)


def load_training_checkpoint(path):
    """Load the new HF directory; retain read compatibility with old .pt exports."""
    path = Path(path)
    if path.is_file():
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = json.loads((path/'training_state.json').read_text())
    index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for filename in sorted(set(index.values())):
        tensors.update(load_file(str(path/filename)))
    if set(tensors) != set(index):
        raise ValueError('Incomplete HF checkpoint tensor inventory')
    state['model'] = tensors
    return state
