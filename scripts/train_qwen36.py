"""Isolated two-stage RGB/MoE trainer. Production launches have no trial preflight."""
from __future__ import annotations
from timejev.config import read_config

import argparse
import contextlib
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist

from timejev.engine import atomic_save, make_loader, save_json, sha256
from timejev.multiturn_data import MultiturnPackedBatchSampler
from timejev.qwen36_checkpoint import (export_checkpoint, find_latest_complete_checkpoint,
                                       load_checkpoint, save_checkpoint)
from timejev.qwen36_model import build_frozen_vision, build_grounder, model_batch, validate_recipe
from timejev.qwen36_data import collator, verify_data as prepare, training_dataset
from timejev.skip_full_safety import PackedUpdateView


def setup(seed):
    import deepspeed
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    # Long RGB videos and metadata can finish unevenly across ranks. This is
    # only a watchdog, not an extra startup communication test.
    dist.init_process_group('nccl', timeout=timedelta(seconds=3600),
                            device_id=torch.device('cuda', local))
    # zero.Init reads DeepSpeedConfig before initializing its own comm wrapper.
    # Bind it to the existing torch group now, otherwise the config silently
    # falls back to world_size=1 and rejects the real global batch size.
    deepspeed.init_distributed(dist_backend='nccl', dist_init_required=False,
                              timeout=timedelta(seconds=3600))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    return rank, world, torch.device('cuda', local)


def schedule_factor(step, total_steps, warmup):
    return ((step + 1) / warmup if step < warmup else
            .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total_steps - warmup)))))


from timejev.qwen36_recipe import parameter_groups


def load_initial_head(model, path, config):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    validate_recipe(saved['config'])
    if saved['config']['training_stage'] != 'head_only':
        raise ValueError('Joint initialization requires a head-only stage handoff')
    expected = ('base', 'model_revision', 'head_dim', 'head_similarity', 'head_logit_scale',
                'readout_mode', 'seed', 'training_split_report', 'max_sequence_length',
                'input_mode', 'rgb_metadata', 'global_batch_size', 'training_tasks')
    if any(saved['config'].get(key) != config.get(key) for key in expected):
        raise ValueError('Head handoff belongs to another model/readout configuration')
    if saved.get('split_sha256') != sha256(config['training_split_report']):
        raise ValueError('Head handoff uses a different complete-video training split')
    if saved.get('planned_steps', 0) <= 0 or saved['step'] != saved['planned_steps']:
        raise ValueError('Head handoff has not completed its planned optimizer updates')
    if saved['decision_metadata'] != model.decision_metadata or not saved.get('complete'):
        raise ValueError('Head handoff is incomplete or has different decision token IDs')
    # Validate every head tensor before token-row loading mutates the target.
    parameters = dict(model.head.named_parameters())
    state = model.head.state_dict()
    if set(saved['head']) != set(state):
        raise ValueError('Head handoff has different parameter keys')
    for key, tensor in saved['head'].items():
        shape = tuple(getattr(parameters.get(key), 'ds_shape', state[key].shape))
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != state[key].dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid head handoff tensor: {key}')
    params = list(model.head.parameters())
    sharded = any(hasattr(p, 'ds_id') for p in params)
    if sharded:
        import deepspeed
    context = deepspeed.zero.GatheredParameters(params, modifier_rank=0) if sharded else contextlib.nullcontext()
    with context:
        if not sharded or dist.get_rank() == 0:
            model.head.load_state_dict(saved['head'], strict=True)
    report = dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=sha256(path),
                  training_seconds=saved['training_seconds'], steps=saved['step'])
    return report


def save_head(model, out, config, step, training_seconds, total_steps):
    import deepspeed
    params = list(model.head.parameters())
    with deepspeed.zero.GatheredParameters(params):
        if dist.get_rank() == 0:
            payload = dict(complete=True, head={k: v.detach().cpu().clone()
                for k, v in model.head.state_dict().items()}, config=config,
                decision_metadata=model.decision_metadata, step=step, planned_steps=total_steps,
                split_sha256=sha256(config['training_split_report']),
                training_seconds=training_seconds)
            atomic_save(payload, out / 'head.pt')
            save_json(dict(complete=True, steps=step, training_seconds=training_seconds,
                           benchmark_deferred=True), out / 'head_stage_complete.json')
    dist.barrier()


def _resume_log(out, rank, saved_step):
    path = out / f'train_rank{rank}.jsonl'
    if not path.exists():
        if saved_step:
            raise ValueError('Resume checkpoint has no corresponding training progress log')
        return
    lines = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row['step'] <= saved_step:
            lines.append((row['step'], line))
    if [x[0] for x in lines] != list(range(1, saved_step + 1)):
        raise ValueError('Resume logs do not cover every completed optimizer update')
    original = path.read_bytes()
    kept = ''.join(line + '\n' for _, line in lines).encode()
    if original != kept:
        folder = out / 'resume_history'
        folder.mkdir(exist_ok=True)
        (folder / f'{time.time_ns()}-{path.name}').write_bytes(original)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(kept)
        temporary.replace(path)


def _complete(path):
    try:
        return json.loads(Path(path).read_text()).get('complete') is True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def stage_complete(out, stage):
    out = Path(out)
    if stage == 'head_only':
        return (_complete(out / 'head_stage_complete.json')
                and (out / 'head.pt').is_file() and (out / 'head.pt').stat().st_size > 0)
    return (_complete(out / 'completion.json')
            and _complete(out / 'hf_checkpoint' / 'complete_manifest.json'))


def check_previous_run(out, config, manifests, world):
    """Check small recorded identities, without loading models or media."""
    path = Path(out) / 'training_config.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous['config'] != config or previous['manifest_hashes'] != manifests
                or previous['world_size'] != world):
            raise ValueError('Existing run has different configuration/data/parallel size; use a new output directory')


def main():
    import deepspeed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', default='auto')
    parser.add_argument('--init-head')
    args = parser.parse_args()
    config = read_config(args.config)
    validate_recipe(config)
    rank, world, device = setup(config['seed'])
    if world != config['expected_world_size'] or config['global_batch_size'] % world:
        raise ValueError('This recipe requires 8 nodes x 8 GPUs and GBS64 packed sequences')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = config['training_stage']
    if stage_complete(out, stage):
        if rank == 0:
            print(json.dumps(dict(event='already_complete', stage=stage)), flush=True)
        dist.destroy_process_group()
        return
    if args.resume not in ('auto', 'none', 'false'):
        if Path(args.resume).resolve() != out / 'safety_deepspeed':
            raise ValueError('Resume must be auto or this run safety_deepspeed directory')
    resume = None if args.resume in ('none', 'false') else find_latest_complete_checkpoint(out / 'safety_deepspeed')
    if rank == 0:
        prepare(config)
    dist.barrier()
    dataset = training_dataset(config)
    pack_spec = json.loads(Path(config['packed_train_plan']).read_text())
    if pack_spec['length_manifest_sha256'] != sha256(config['packed_train_lengths']):
        raise ValueError('RGB packing plan and episode lengths disagree')
    manifests = {path: sha256(path) for path in [config['train'], config['packed_train_lengths'],
                                               config['packed_train_plan'], *config['benchmarks'].values()]}
    rgb_identity = [{path: sha256(path) for path in
        (config['rgb_metadata'], config['rgb_ready'], config['training_split_report'])} if rank == 0 else None]
    dist.broadcast_object_list(rgb_identity, src=0)
    manifests.update(rgb_identity[0])
    check_previous_run(out, config, manifests, world)
    if resume is None and rank == 0 and any(p.stat().st_size for p in out.glob('train_rank*.jsonl')):
        # No complete save means no optimizer progress can be recovered. The
        # same command remains restartable, but never claims to have resumed.
        print(json.dumps(dict(event='restart_from_step_zero', stage=stage,
            reason='No complete optimizer checkpoint; original stage initialization will be used',
            previous_logs='archived in resume_history')), flush=True)
    sampler = MultiturnPackedBatchSampler(dataset.lengths, config['max_sequence_length'], rank, world,
        config['seed'], config['global_batch_size'], episodes=dataset.episodes, packs=pack_spec['packs'])
    total_steps = sampler.total_updates * config['epochs']
    warmup = max(1, round(total_steps * config['warmup_ratio']))
    ds_config = json.loads(Path(config['deepspeed']).read_text())
    accumulation = config['global_batch_size'] // world
    ds_config.update(train_batch_size=config['global_batch_size'], train_micro_batch_size_per_gpu=1,
                     gradient_accumulation_steps=accumulation, gradient_clipping=config['grad_clip'])
    print(json.dumps(dict(event='initializing_sharded_model', rank=rank, stage=stage,
                          resume=resume[1] if resume else None)), flush=True)
    # Joint checkpoints contain every trainable language/head parameter. A
    # resumed head-only checkpoint excludes its unchanged frozen language.
    model = build_grounder(config, device, ds_config=ds_config,
                           load_pretrained=not (resume and stage == 'joint'))
    initialization = None
    if stage == 'joint' and resume is None:
        if not args.init_head:
            raise ValueError('Fresh joint stage requires this experiment completed head stage')
        initialization = load_initial_head(model, args.init_head, config)
    groups = parameter_groups(model, config)
    base_lrs = [g['lr'] for g in groups]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, fused=True)
    engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config,
                                                  dist_init_required=False)
    visual = build_frozen_vision(config, device)
    state = {}
    if resume:
        state = load_checkpoint(engine, resume[0], tag=resume[1])
        if state['config'] != config or state['manifest_hashes'] != manifests or state['world_size'] != world:
            raise ValueError('Resume configuration/data/parallel size does not match the saved run')
        initialization = state.get('initialization')
    step = int(state.get('step', 0))
    if not 0 <= step <= total_steps:
        raise ValueError('Resume progress is outside this stage update plan')
    training_seconds = float(state.get('training_seconds', 0.0))
    _resume_log(out, rank, step)
    identities = [state.get('checkpoint_id') or str(uuid.uuid4()) if rank == 0 else None]
    dist.broadcast_object_list(identities, src=0)
    checkpoint_id = identities[0]

    def set_lr(index):
        factor = schedule_factor(index, total_steps, warmup)
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = factor * lr

    set_lr(step)
    if rank == 0:
        from timejev.decision_inputs import load_processor
        save_json(dict(config=config, world_size=world, total_steps=total_steps,
            total_epochs=config['epochs'], visual_protocol=model.visual_protocol,
            trainable_parameters=sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters() if p.requires_grad),
            unique_training_queries=len(dataset.rows), packed_training_sequences=len(sampler.packs),
            initialization=initialization, manifest_hashes=manifests,
            batch_semantics='64 packed sequences per update, each with one or more complete video conversations; query-normalized loss; no repeated tail samples',
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                         host=socket.gethostname(), experts='XTuner Triton grouped GEMM', zero_stage=3,
                         token_dispatch='XTuner CUDA permute/unpermute', router_compute_dtype='float32',
                         deepspeed=deepspeed.__version__,
                         parameter_dtype='float32', compute_dtype='bfloat16', vision='replicated frozen RGB')), out / 'training_config.json')
        save_json(model.decision_metadata, out / 'decision_metadata.json')
        load_processor(config['base']).tokenizer.save_pretrained(out / 'tokenizer')
        save_json(dict(status='训练中', message=f'Qwen3.6 RGB {stage}; completed {step}/{total_steps}'), out / 'status.json')
        resume_root = out / 'safety_deepspeed'
        resume_root.mkdir(exist_ok=True)
        save_json(dict(format='timejev_qwen36_resume_v1', run=str(out)), resume_root / '.timejev_qwen36_owner.json')
    dist.barrier()
    if step < total_steps:
        view = PackedUpdateView(sampler, step + 1) if step else sampler
        loader = make_loader(dataset, collator(config, training=True), config['workers'], batch_sampler=view)
        iterator = iter(loader)
        # Starting a DataLoader consumes RNG for worker seeds. Restore the
        # checkpoint RNG afterwards to continue optimizer/model randomness.
        if state.get('rng_state_by_rank'):
            from timejev.qwen36_checkpoint import _restore_rng
            _restore_rng(state['rng_state_by_rank'][rank])
        engine.train()
        accumulated = 0.0
        counts = [0, 0, 0, 0, 0]
        previous = time.monotonic()
        with (out / f'train_rank{rank}.jsonl').open('a', buffering=1) as log:
            for batch in iterator:
                schedule = batch['pack_schedule']
                boundary = schedule[1] + 1 == schedule[2]
                engine.set_gradient_accumulation_boundary(boundary)
                weight = batch['sample_count'] * world / schedule[3]
                aux_scope = contextlib.nullcontext()
                with aux_scope as aux:
                    output = engine(**model_batch(batch, visual, device))
                    loss = output['loss']
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at scheduled update {schedule[0]}')
                    engine.backward(loss * weight * accumulation)
                engine.step()
                accumulated += float(loss.detach()) * weight
                counts[0] += batch['sample_count']
                counts[1] += batch['real_tokens']
                counts[2] += int(batch['sample_count'] > 0)
                counts[3] += int(output['task_counts'][0])
                counts[4] += int(output['task_counts'][1])
                if not boundary:
                    continue
                torch.cuda.synchronize()
                elapsed = time.monotonic() - previous
                training_seconds += elapsed
                step += 1
                set_lr(step)
                scalar = torch.tensor(accumulated, device=device)
                count = torch.tensor(counts, device=device, dtype=torch.long)
                dist.all_reduce(scalar)
                dist.all_reduce(count)
                if int(count[0]) != schedule[3] or int(count[2]) != schedule[4]:
                    raise ValueError('Global query count or packed batch normalization changed')
                norm = engine.get_global_grad_norm()
                if norm is not None and not math.isfinite(float(norm)):
                    raise FloatingPointError(f'Nonfinite gradient norm at step {step}')
                record = dict(epoch=1, step=step, loss=float(scalar) / world,
                    grad_norm=float(norm) if norm is not None else None, next_lr=optimizer.param_groups[0]['lr'],
                    elapsed=training_seconds, update_seconds=elapsed, global_samples=int(count[0]),
                    global_real_tokens=int(count[1]), global_packed_sequences=int(count[2]),
                    vtg_queries=int(count[3]), qa_queries=int(count[4]),
                    effective_tokens_per_second=int(count[1]) / elapsed,
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
                log.write(json.dumps(record) + '\n')
                if rank == 0:
                    print(json.dumps(record), flush=True)
                if step % config.get('save_every_steps', 100) == 0 or step == total_steps:
                    timing = torch.tensor(training_seconds, device=device)
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    training_seconds = float(timing)
                    save_checkpoint(engine, out / 'safety_deepspeed', step, dict(
                        step=step, next_update=step + 1, next_epoch=0, config=config,
                        world_size=world, manifest_hashes=manifests, checkpoint_id=checkpoint_id,
                        initialization=initialization, training_seconds=training_seconds))
                previous = time.monotonic()
                accumulated, counts = 0.0, [0, 0, 0, 0, 0]
    if step != total_steps:
        raise RuntimeError(f'Training iterator ended at {step}/{total_steps} updates')
    if stage == 'head_only':
        save_head(model, out, config, step, training_seconds, total_steps)
        if rank == 0:
            save_json(dict(complete=True, steps=step, planned_steps=total_steps,
                           training_seconds=training_seconds, benchmark_seconds=0.0,
                           benchmark_deferred=True), out / 'completion.json')
            save_json(dict(status='阶段完成', message='30% VTG BCE / QA CE warmup completed; continue joint stage'), out / 'status.json')
    else:
        # A complete standalone model is durable before benchmark starts. A
        # failed evaluator never requires retraining the completed last update.
        export_checkpoint(model, out / 'hf_checkpoint', config, checkpoint_id, step)
        if rank == 0:
            save_json(dict(complete=True, steps=step, planned_steps=total_steps,
                training_seconds=training_seconds, benchmark_seconds=0.0, benchmark_deferred=True,
                checkpoint_id=checkpoint_id, model_directory=str(out / 'hf_checkpoint'),
                initialization=initialization,
                total_two_stage_training_seconds=training_seconds + (initialization or {}).get('training_seconds', 0.0)),
                out / 'completion.json')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
