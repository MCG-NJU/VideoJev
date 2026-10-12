"""30% head warmup then remaining 70% joint VTG/QA training.

Uses the final VTG model/optimizer with task-specific auxiliary losses in stage 2.
Saves native DeepSpeed checkpoints independently of benchmark execution.
"""
from timejev.config import read_config
import argparse
import hashlib
import json
import importlib.metadata
import math
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
import torch
import torch.distributed as dist
from timejev.benchmark import evaluate_benchmarks
from timejev.data import Collator, GroundingDataset, to_device
from timejev.engine import BucketSampler, atomic_save, make_loader, save_json, setup, sha256
from timejev.vision import VISUAL_PROTOCOL, load_processor
from timejev.skip_full_safety import (PackedUpdateView, capture_rng, restore_rng,
    save_safety_checkpoint, verify_checkpoint_inventory, training_offsets)
from timejev.skip_full_safety import state_fingerprint, archive_replayed_progress


def link_checkpoint(source, link):
    temp = link.with_suffix('.tmp')
    temp.unlink(missing_ok=True)
    temp.symlink_to(source.name)
    os.replace(temp, link)


def parse_arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--resume', help='DeepSpeed checkpoint directory')
    p.add_argument('--init-head', help='Completed head-only stage; start a fresh joint optimizer/scheduler')
    p.add_argument('--diagnostic-updates',type=int,default=0,
                   help='Stop a packed training-path check after N real updates; no benchmark/checkpoint or formal result')
    p.add_argument('--diagnostic-tail-updates',type=int,default=0)
    p.add_argument('--diagnostic-checkpoint',action='store_true',
                   help='Exercise native save/load and continuation between the two prefix diagnostic updates')
    a = p.parse_args(argv)
    # The suite's existing two-update prefix command also exercises protection;
    # a duplicate pair of expensive 144K updates is unnecessary.
    a.diagnostic_checkpoint = a.diagnostic_checkpoint or a.diagnostic_updates == 2
    return a


def validate_recipe(c):
    head = c.get('training_stage') == 'head_only'
    rgb = c.get('input_mode') == 'rgb_dynamic'
    expected = dict(formulation='timejev_decision_v1', shared_decision_head=True,
        experiment_arch='timestamp_option_memory', grounding_candidates='timestamp_options',
        training_layout='shared_video_multiturn_v1', readout_mode='last_attention',
        attention_mode='causal', head_dim=512, head_similarity='cosine', head_logit_scale=10.,
        freeze_visual=True, freeze_llm=head, checkpointing=not head, epochs=1,
        lr=0. if head else 5e-6, head_lr=3e-4 if head else 1e-4,
        grounding_loss='bce' if head else 'bce_soft_iou', soft_iou_weight=0. if head else 2.,
        global_batch_size=64,
        max_sequence_length=196608 if rgb else 147456, train_fps=2., eval_fps=4.,
        sequence_packing=True, pack_to_max_length=True)
    if c.get('training_stage') not in ('head_only', 'joint'):
        raise ValueError('An explicit head_only/joint stage is required')
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f'Confirmed recipe differs in {key}: {c.get(key)!r} != {value!r}')


def main():
    import deepspeed
    from timejev.model import from_config
    from timejev.data import Collator, GroundingDataset
    from timejev.vision import load_processor
    from timejev.benchmark import evaluate_benchmarks
    a = parse_arguments()
    c = read_config(a.config)
    if a.init_head:
        c['initial_head_checkpoint'] = str(Path(a.init_head).resolve())
    validate_recipe(c)
    rgb = c.get('input_mode') == 'rgb_dynamic'
    if rgb:
        # ZeRO-3 reload needs a fresh engine. Its checkpoint roundtrip is tested
        # in separate processes, rather than corrupting/reloading this engine.
        a.diagnostic_checkpoint = False
    if bool(c.get('initial_head_checkpoint')) != (c.get('training_stage') == 'joint'):
        raise ValueError('Joint stage requires --init-head; other stages start from the base model')
    diagnostic_updates=a.diagnostic_updates or a.diagnostic_tail_updates
    if (min(a.diagnostic_updates,a.diagnostic_tail_updates)<0 or
            (a.diagnostic_updates and a.diagnostic_tail_updates) or
            (a.diagnostic_checkpoint and (a.diagnostic_updates!=2 or a.diagnostic_tail_updates))):
        raise ValueError('Invalid independent diagnostic request')
    decision = bool(c.get('formulation'))
    if decision:
        from timejev.decision_model import from_config
        from timejev.timestamp_option_inputs import load_processor
        from timejev.decision_io import data_module
        decision_data = data_module(c)
        from timejev.multiturn_data import MultiturnDataset as GroundingDataset, MultiturnCollator as Collator
        from timejev.decision_benchmark import evaluate_benchmarks
    from timejev.feature_sources import feature_sources, register_training_caches
    train_feature_root, train_variant_root = feature_sources(c, training=True)
    packing=c.get('sequence_packing',False)
    global_batch=c.get('global_batch_size',32)
    if diagnostic_updates and (not packing or a.resume):
        raise ValueError('Prefix diagnostics use new packed runs only')
    if c.get('feature_root'):
        from timejev.features import validate_cache
        from timejev.multiturn_data import training_cache_specs
        train_specs=training_cache_specs(c)
        eval_spec=validate_cache(c['feature_root'], c['benchmarks'].values(), c['eval_fps'],replacement_root=c.get('feature_variant_root'))
        assert all(s['vision_sha256']==eval_spec['vision_sha256'] for s in train_specs), 'Training and benchmark ViT weights differ'
    elif c.get('eval_archive_root'):
        from timejev.frame_archive import validate_index
        validate_index(c['eval_archive_root'], c['benchmarks'].values(), c['eval_fps'])
    rank, world, device = setup(c['seed'])
    if packing:
        if (c.get('global_batch_unit')!='packed_sequence' or global_batch not in (16,32,64,128)
                or global_batch%world or c['batch_size']!=1 or c['accumulation']!=global_batch//world):
            raise ValueError('Global batch counts packed sequences; accumulation=global_batch/world')
        if not c.get('freeze_visual') or not (c.get('feature_root') or rgb):
            raise ValueError('Packed training requires frozen visual features')
    if world != c['expected_world_size']:
        raise ValueError('This configuration must run with its explicit world size')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume and (out / 'training_config.json').exists():
        raise FileExistsError('Use a new output directory or explicitly --resume')
    from timejev.multiturn_data import training_dataset
    dataset = training_dataset(c)
    collator_options = ({'include_score_durations': True}
                        if decision and c.get('grounding_loss') == 'bce_soft_iou'
                        and c.get('soft_iou_weight', 0) > 0 else {})
    if packing:
        from timejev.multiturn_data import MultiturnPackedBatchSampler as PackedBatchSampler, MultiturnCollator as PackedCollator
        from timejev.packing import xtuner_source_hashes
        pack_spec=json.loads(Path(c['packed_train_plan']).read_text())
        assert pack_spec['length_manifest_sha256']==sha256(c['packed_train_lengths'])
        assert pack_spec['xtuner_source_sha256']==xtuner_source_hashes()
        assert pack_spec['seed']==c['seed'] and pack_spec['max_sequence_length']==c['max_sequence_length']
        sampler=PackedBatchSampler(dataset.lengths,c['max_sequence_length'],rank,world,c['seed'],global_batch,
                                   packs=pack_spec['packs'],episodes=dataset.episodes)
        if a.diagnostic_tail_updates:
            sampler=PackedUpdateView(sampler,sampler.total_updates-a.diagnostic_tail_updates+1)
        collator=PackedCollator(load_processor(c['base']).tokenizer,c['max_sequence_length'],pad_to_max_length=True,
                               **collator_options)
    else:
        sampler = BucketSampler(dataset.lengths, c['batch_size'], rank, world, c['seed'],
                                reference_world=c.get('sampler_reference_world'))
        collator=Collator(load_processor(c['base']).tokenizer, **collator_options)
    loader = make_loader(dataset,collator,c['workers'],batch_sampler=sampler)
    model = from_config(c).to(device)
    initialization = None
    if c.get('initial_head_checkpoint'):
        from timejev.staged_training import initial_head
        initialization = initial_head(model, c['initial_head_checkpoint'], c)
    if decision and rank == 0:
        load_processor(c['base']).tokenizer.save_pretrained(out / 'tokenizer')
        save_json(model.decision_metadata, out / 'decision_metadata.json')
    if c.get('feature_root'):
        register_training_caches(model, *train_specs)
    # No weight decay on biases, norms, or scalar/vector dynamics parameters.
    from timejev.staged_training import optimizer_groups, lr_factor
    groups = optimizer_groups(model, c)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, fused=True)
    steps_per_epoch = sampler.total_updates if packing else math.ceil(len(loader)/c['accumulation'])
    total_steps = steps_per_epoch * c['epochs']
    base_lrs = [g['lr'] for g in groups]
    dsconfig = json.loads(Path(c['deepspeed']).read_text())
    if rgb:
        if dsconfig['zero_optimization']['stage'] != 3:
            raise ValueError('The RGB 192K recipe requires ZeRO-3')
        from timejev.rgb_model import configure_zero3
        configure_zero3(model)
    dsconfig.update(train_micro_batch_size_per_gpu=c['batch_size'],
                    gradient_accumulation_steps=c['accumulation'],
                    train_batch_size=c['batch_size'] * c['accumulation'] * world,
                    gradient_clipping=c['grad_clip'])
    engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=dsconfig,
                                                  dist_init_required=not dist.is_initialized())
    assert all(p.dtype == torch.float32 for p in engine.module.parameters())
    assert engine.module.backbone.config.vision_config._attn_implementation == 'flash_attention_2'
    start_epoch, step, best = 0, 0, -1.
    if a.diagnostic_tail_updates:
        step=steps_per_epoch-a.diagnostic_tail_updates
    diagnostic_start_step=step
    offsets=dict(elapsed=0.,training_seconds=0.,benchmark_seconds=0.)
    resume_rng=None
    provenance_paths = [c['train'], *c['benchmarks'].values()]
    if c.get('feature_root'):
        provenance_paths += [str(Path(c['feature_root']) / name) for name in ['provenance.json','index.json']]
    if c.get('feature_variant_root'):
        provenance_paths += [str(Path(c['feature_variant_root'])/name) for name in ['provenance.json','index.json','resolution_plan.json']]
    if c.get('train_feature_root'):
        provenance_paths += [str(Path(c['train_feature_root'])/name) for name in ['provenance.json','index.json']]
    if c.get('train_feature_variant_root'):
        provenance_paths += [str(Path(c['train_feature_variant_root'])/name) for name in ['provenance.json','index.json','resolution_plan.json']]
    if c.get('training_sources'):
        from timejev.timestamp_mixture_data import provenance_paths as mixture_provenance_paths
        provenance_paths += mixture_provenance_paths(c)
    if c.get('training_filter_report'):
        provenance_paths.append(c['training_filter_report'])
    for key in ('training_split_report', 'initial_head_checkpoint'):
        if c.get(key):
            provenance_paths.append(c[key])
    if packing:
        provenance_paths += [c['packed_train_lengths'],c['packed_train_plan'],*c.get('packed_eval_lengths',{}).values()]
    if rgb:
        provenance_paths += [c['rgb_metadata'], c['rgb_evaluation'], c['rgb_ready']]
    if dataset.metadata_path.exists():
        provenance_paths.append(str(dataset.metadata_path))
    hashes = {path: sha256(path) for path in provenance_paths}
    if a.resume:
        inventory=verify_checkpoint_inventory(a.resume)
        loaded, state = engine.load_checkpoint(a.resume, tag=inventory['tag'], load_module_strict=True)
        if loaded is None or state['config'] != c or state['world_size'] != world or state['manifest_hashes'] != hashes:
            raise ValueError('Resume checkpoint/config/world/data mismatch')
        assert state['visual_protocol'] == VISUAL_PROTOCOL
        start_epoch, step, best = state['next_epoch'], state['step'], state['best_benchmark']
        if not state.get('safety_checkpoint') or state['saved_step']!=step:
            raise ValueError('Resume requires this run\'s complete safety checkpoint')
        archive_replayed_progress(out,saved_step=step,rank=rank,world=world)
        offsets=training_offsets(state)
        offsets['elapsed']+=inventory['save_seconds']
        offsets['training_seconds']+=inventory['save_seconds']
        resume_rng=state['rng_state_by_rank'][rank]
        if state['next_update']<=steps_per_epoch:
            sampler=PackedUpdateView(sampler,state['next_update'])
            loader=make_loader(dataset,collator,c['workers'],batch_sampler=sampler)
        else:
            # A completed-training checkpoint can retry benchmark without
            # repeating any sample or optimizer update.
            loader=[]
    def set_lr(index):
        factor = lr_factor(index, total_steps, c['warmup_ratio'])
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = factor * lr
    set_lr(step)
    if rank == 0:
        save_json(dict(config=c, deepspeed=dsconfig, visual_protocol=VISUAL_PROTOCOL, world_size=world,
                       experiment_kind=('packed_training_tail_diagnostic' if a.diagnostic_tail_updates else
                           'packed_training_prefix_diagnostic') if diagnostic_updates else 'formal_training',
                       formal_training=not bool(diagnostic_updates),
                       diagnostic_first_update=diagnostic_start_step+1 if diagnostic_updates else None,
                       diagnostic_checkpoint_validation=a.diagnostic_checkpoint,
                       safety_checkpoint_policy='every100_updates_and_before_benchmark_keep_latest2',
                       trainable_parameters=sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters() if p.requires_grad), total_steps=total_steps,
                       initialization=initialization,
                       unique_training_queries=len(dataset.rows),
                       training_layout=c['training_layout'],training_episodes=len(dataset),
                       packed_training_sequences=len(sampler.packs) if packing else None,
                       xtuner_source_sha256=xtuner_source_hashes() if packing else None,
                       runtime={**{name: importlib.metadata.version(name) for name in
                                   ['torch','transformers','deepspeed','flash-attn','flash-linear-attention','causal-conv1d','av','zstandard']},
                                'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
                                'host': socket.gethostname(),
                                'nnodes': world//int(os.environ.get('LOCAL_WORLD_SIZE',world)),
                                'node_rank': os.environ.get('GROUP_RANK'),
                                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                                'fla_intracard_cp': os.environ.get('FLA_INTRACARD_CP'),
                                'input_transfer': ('two pinned RGB chunks with H2D prefetch; CPU worker preprocessing'
                                                   if rgb else 'two 128 MiB pinned buffers; exact full-tensor H2D')},
                       repeated_padding_queries=(0 if packing else len(sampler)*c['batch_size']*world-len(dataset)),
                       batch_semantics=f'{global_batch} packed sequences/update; each nonempty sequence padded to {c["max_sequence_length"]}; per-query loss equally weights original queries in the update; final partial packs kept without repeats' if packing else 'one sample per micro-batch',
                       manifest_hashes=hashes,
                       code_sha256={str(p): sha256(p) for root in ['timejev', 'scripts', 'configs']
                                    for p in Path(root).glob('*') if p.suffix in ['.py', '.json']}),
                  out / 'training_config.json')
        save_json(dict(status='训练中', message=('stage 1: head only; entire backbone frozen' if c.get('freeze_llm') else
                  'stage 2: joint LLM/head training after head warmup' if initialization else
                  'native backbone training; visual encoder and merger frozen'
                  if c.get('freeze_visual') else 'native full-parameter training')), out / 'status.json')
    started = time.monotonic()
    training_seconds, benchmark_seconds = offsets['training_seconds'], offsets['benchmark_seconds']
    checkpoint_reports=[]
    with open(out / f'train_rank{rank}.jsonl', 'a', buffering=1) as log:
        for epoch in range(start_epoch, c['epochs']):
            sampler.epoch = epoch
            engine.train()
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
            epoch_start = time.monotonic()
            previous_update_end = epoch_start
            accumulated = 0.
            query_count=real_tokens=packed_tokens=pack_calls=real_packs=episode_count=0
            loss_sums = torch.zeros(3, device=device)
            task_counts = torch.zeros(2, device=device, dtype=torch.long)
            local_query_ids=[]
            epoch_loader=iter(loader)
            if resume_rng is not None:
                restore_rng(resume_rng);resume_rng=None
            for micro, batch in enumerate(epoch_loader):
                if rank == 0 and (micro < 4 or micro % 40 == 0):
                    print(json.dumps(dict(event='micro_start', epoch=epoch + 1, micro=micro,
                                          tokens=int(batch['input_ids'].shape[1]),
                                          score_points=int(batch['score_mask'].sum()),
                                          elapsed=time.monotonic() - started)), flush=True)
                boundary = (batch['pack_schedule'][1]+1==batch['pack_schedule'][2] if packing else
                            (micro+1)%c['accumulation']==0 or micro+1==len(loader))
                group_size = min(c['accumulation'], len(loader) - micro // c['accumulation'] * c['accumulation'])
                engine.set_gradient_accumulation_boundary(boundary)
                output = engine(**to_device(batch, device))
                loss = output['loss']
                loss_sums += output['loss_sums']
                task_counts += output['task_counts']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss: epoch {epoch}, micro {micro}')
                # DeepSpeed divides by configured accumulation even at the final partial group.
                if packing:
                    weight=batch['sample_count']*world/batch['pack_schedule'][3]
                    engine.backward(loss*weight*c['accumulation'])
                    accumulated+=float(loss.detach())*weight
                    query_count+=batch['sample_count'];real_tokens+=batch['real_tokens']
                    packed_tokens+=batch['input_ids'].shape[1];pack_calls+=1
                    real_packs+=int(batch['sample_count']>0)
                    episode_count+=batch['episode_count']
                    local_query_ids.extend(row['id'] for row in batch['rows'])
                else:
                    engine.backward(loss * c['accumulation'] / group_size)
                    accumulated += float(loss.detach()) / group_size
                engine.step()
                if not boundary:
                    continue
                step += 1
                set_lr(step)
                mean_loss = torch.tensor(accumulated, device=device)
                if world > 1:
                    dist.all_reduce(mean_loss)
                norm = engine.get_global_grad_norm()
                if norm is not None and not math.isfinite(float(norm)):
                    raise FloatingPointError(f'Nonfinite gradient norm at optimizer step {step}')
                if world > 1:
                    dist.all_reduce(loss_sums)
                    dist.all_reduce(task_counts)
                record = dict(epoch=epoch + 1, step=step, loss=float(mean_loss) / world,
                              grad_norm=float(norm) if norm is not None else None,
                              next_lr=optimizer.param_groups[0]['lr'],
                              next_head_lr=optimizer.param_groups[-1]['lr'],
                              elapsed=offsets['elapsed']+time.monotonic() - started,
                              peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                              peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
                for name, value, count in zip(
                    ('vtg_bce', 'qa_ce', 'vtg_soft_iou'),
                    loss_sums.tolist(), task_counts[[0, 1, 0]].tolist()):
                    record[name] = value / max(1, count)
                record['vtg_queries'], record['qa_queries'] = task_counts.tolist()
                loss_sums.zero_(); task_counts.zero_()
                if packing:
                    counts=torch.tensor([query_count,real_tokens,packed_tokens,real_packs,episode_count],device=device,dtype=torch.long)
                    dist.all_reduce(counts) if world>1 else None
                    assert int(counts[0])==batch['pack_schedule'][3]
                    assert int(counts[3])==batch['pack_schedule'][4]
                    record.update(global_samples=int(counts[0]),global_real_tokens=int(counts[1]),
                                  global_padded_tokens=int(counts[2]),global_packed_sequences=int(counts[3]),
                                  micro_rounds=pack_calls,global_video_episodes=int(counts[4]),
                                  local_query_ids_sha256=hashlib.sha256(json.dumps(local_query_ids).encode()).hexdigest())
                    if a.diagnostic_tail_updates:
                        record.update(scheduled_update=batch['pack_schedule'][0],
                                      last_micro_local_samples=batch['sample_count'])
                    now=time.monotonic()
                    record.update(update_seconds=now-previous_update_end,
                                  effective_tokens_per_second=int(counts[1])/(now-previous_update_end),
                                  queries_per_second=int(counts[0])/(now-previous_update_end))
                    previous_update_end=now
                    query_count=real_tokens=packed_tokens=pack_calls=real_packs=episode_count=0
                    local_query_ids=[]
                log.write(json.dumps(record) + '\n')
                if rank == 0 and (step <= 3 or step % 10 == 0):
                    print(json.dumps(record), flush=True)
                accumulated = 0.
                save_now=((not diagnostic_updates and (step%100==0 or step==total_steps)) or
                          (a.diagnostic_checkpoint and step==1))
                if save_now:
                    report=save_safety_checkpoint(engine,out/'safety_deepspeed',rank=rank,world=world,step=step,
                        client_state=dict(next_epoch=epoch,next_update=batch['pack_schedule'][0]+1,
                            step=step,best_benchmark=best,config=c,world_size=world,visual_protocol=VISUAL_PROTOCOL,
                            manifest_hashes=hashes,timing_offsets=dict(
                                elapsed=offsets['elapsed']+time.monotonic()-started,
                                training_seconds=training_seconds+time.monotonic()-epoch_start,
                                benchmark_seconds=benchmark_seconds)))
                    checkpoint_reports.append(report)
                    if a.diagnostic_checkpoint:
                        # Check real native model/optimizer reload between two
                        # actual 144K updates, with the original RNG retained.
                        rng=capture_rng()
                        before_steps=engine.global_steps
                        before_head={k:v.detach().clone() for k,v in model.head.state_dict().items()}
                        before_optimizer=state_fingerprint(engine.optimizer.state_dict())
                        with torch.no_grad():
                            next(model.head.parameters()).reshape(-1)[0].add_(.125)
                            states=engine.optimizer.optimizer.state
                            changed_optimizer=False
                            for value in states.values():
                                if 'exp_avg' in value:
                                    value['exp_avg'].reshape(-1)[0].add_(.25)
                                    if torch.is_tensor(value.get('step')):
                                        value['step'].add_(1)
                                    elif 'step' in value:
                                        value['step']+=1
                                    changed_optimizer=True
                                    break
                        assert changed_optimizer and state_fingerprint(engine.optimizer.state_dict())!=before_optimizer
                        started_load=time.monotonic()
                        inventory=verify_checkpoint_inventory(out/'safety_deepspeed')
                        loaded,reloaded=engine.load_checkpoint(str(out/'safety_deepspeed'),tag=inventory['tag'],load_module_strict=True)
                        assert loaded is not None and reloaded['step']==step and engine.global_steps==before_steps
                        for k,v in model.head.state_dict().items():
                            torch.testing.assert_close(v,before_head[k],atol=0,rtol=0)
                        assert state_fingerprint(engine.optimizer.state_dict())==before_optimizer
                        restore_rng(rng)
                        if rank==0:
                            save_json(dict(passed=True,real_world_size=world,save_report=report,
                                load_seconds=time.monotonic()-started_load,head_reload_exact=True,
                                all_rank_optimizer_reload_exact=True,
                                deliberate_head_and_adam_corruption_repaired=True,
                                engine_steps_restored=True),out/'checkpoint_check.json')
                if diagnostic_updates and step-diagnostic_start_step>=diagnostic_updates:
                    break
            torch.cuda.synchronize()
            epoch_time = torch.tensor(time.monotonic() - epoch_start, device=device)
            if world > 1:
                dist.all_reduce(epoch_time, op=dist.ReduceOp.MAX)
            training_seconds += float(epoch_time)
            if diagnostic_updates:
                if step-diagnostic_start_step != diagnostic_updates:
                    raise ValueError('Insufficient data for requested diagnostic updates')
                if rank==0:
                    save_json(dict(complete=False,diagnostic_complete=True,formal_training=False,
                                   steps=step-diagnostic_start_step,final_scheduled_step=step,
                                   diagnostic_first_update=diagnostic_start_step+1,
                                   diagnostic_tail_updates=a.diagnostic_tail_updates,
                                   planned_steps=total_steps,training_seconds=training_seconds),out/'completion.json')
                    save_json(dict(status='训练路径检查完成',message='真实数据和优化器的短程检查；非完整训练，无benchmark成绩'),out/'status.json')
                if dist.is_initialized():dist.destroy_process_group()
                return
            if c.get('training_stage') == 'head_only':
                if rgb:
                    from timejev.rgb_model import consolidated_state
                    exported_head = consolidated_state(model.head)
                if rank == 0:
                    atomic_save(dict(format='timejev_head_stage_v1',
                        head=exported_head if rgb else {k:v.detach().cpu() for k,v in model.head.state_dict().items()},
                        config=c, decision_metadata=model.decision_metadata,
                        split_sha256=sha256(c['training_split_report']), source_run=str(out.resolve()),
                        steps=step, training_queries=len(dataset.rows), training_seconds=training_seconds),
                        out/'head.pt')
                    completion = dict(complete=step==total_steps, steps=step, planned_steps=total_steps,
                        training_seconds=training_seconds, benchmark_seconds=0.,
                        wall_seconds=offsets['elapsed']+time.monotonic()-started,
                        safety_checkpoints=checkpoint_reports, visual_protocol=VISUAL_PROTOCOL,
                        training_stage='head_only', benchmark_deferred_to_joint_stage=True)
                    save_json(completion, out/'completion.json')
                    save_json(dict(status='阶段完成', message='Head warmup complete; BCE/CE only'), out/'status.json')
                    save_json(dict(complete=True, head=str((out/'head.pt').resolve())), out/'head_stage_complete.json')
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                return
            identity = [str(uuid.uuid4()) if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(identity)
            checkpoint_id = identity[0]
            # Publish weights before benchmarks. A failed evaluation can retry
            # directly from this export without restoring optimizer state.
            if rgb:
                from timejev.rgb_model import consolidated_state
                exported_model = consolidated_state(model)
            if rank == 0:
                checkpoint = dict(model=exported_model if rgb else {k:v.detach().cpu() for k,v in model.state_dict().items()},
                    config=c, visual_protocol=VISUAL_PROTOCOL, checkpoint_id=checkpoint_id,
                    epoch=epoch+1, step=step, manifest_hashes=hashes,
                    decision_metadata=model.decision_metadata)
                from timejev.hf_checkpoint import export_training_checkpoint
                export_training_checkpoint(checkpoint, out/'hf_checkpoint')
                del checkpoint
            if world > 1:
                dist.barrier()
    if rank == 0:
        save_json(dict(complete=step == total_steps, steps=step, planned_steps=total_steps,
                       training_seconds=training_seconds, benchmark_seconds=benchmark_seconds,
                       wall_seconds=offsets['elapsed']+time.monotonic() - started,
                       initialization=initialization,
                       total_two_stage_training_seconds=(training_seconds+initialization['training_seconds']
                                                        if initialization else None),
                       safety_checkpoints=checkpoint_reports,visual_protocol=VISUAL_PROTOCOL), out / 'completion.json')
        save_json(dict(status='训练完成，等待七榜评测'), out / 'status.json')
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
