"""Decision-model inference with the unchanged single-interval TimeLens evaluator."""
import json
import contextlib
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from .benchmark import NAMES, select_decoder, export
from .data import to_device
from .decision_data import DecisionDataset, DecisionCollator, PackedDecisionCollator, load_lengths
from .decision_inputs import FORMULATION, CHOICE, task_of, load_processor
from .engine import make_loader, predict, atomic_save, save_json, sha256
from .packing import best_fit
from .vision import VISUAL_PROTOCOL


@torch.inference_mode()
def predict_decisions(model, loader, device, *, threshold=.5, smooth=1):
    """Mixed grounding/QA inference. Targets are never passed into the model."""
    from .temporal import decode
    model.eval()
    results = []
    cache = model.video_cache(max_tokens=147456) if hasattr(model, 'video_cache') else contextlib.nullcontext()
    with cache:
        for batch in loader:
            inputs = to_device(batch, device)
            inputs.pop('labels', None)
            out = model(**inputs)
            logits = out['logits'].float().cpu()
            probabilities = out['probabilities'].float().cpu()
            for i, row in enumerate(batch['rows']):
                n = int(batch['score_mask'][i].sum())
                result = dict(id=row['id'], task=row.get('task','grounding'), logits=logits[i,:n].tolist(),
                              probabilities=probabilities[i,:n].tolist(), formulation=FORMULATION)
                if task_of(row) == CHOICE:
                    answer = int(probabilities[i,:n].argmax())
                    result.update(answer_index=answer, answer=row['options'][answer], options=row['options'])
                else:
                    result.update(times=batch['times'][i].tolist(), interval=decode(logits[i,:n],batch['times'][i],
                        row['sampled_duration'],threshold,smooth,bounds=batch['audits'][i]['score_bounds']))
                results.append(result)
    return results


def evaluate_benchmarks(model, config, output, checkpoint_id, rank, world, device):
    from .decision_io import data_module
    inputs = data_module(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_processor(config['base']).tokenizer
    all_rows, all_scores, timing = {}, {}, {}
    for name in NAMES:
        dataset = inputs.DecisionDataset(config['benchmarks'][name], config['base'], fps=config['eval_fps'],
            query_placement='last', feature_root=config['feature_root'],
            feature_variant_root=config.get('feature_variant_root'), fast_feature_inputs=True)
        if any(task_of(row) == CHOICE for row in dataset.rows):
            raise ValueError('TimeLens benchmarks must contain grounding questions')
        if config.get('sequence_packing'):
            dataset.lengths = inputs.load_lengths(config['packed_eval_lengths'][name],config['benchmarks'][name],config,dataset.rows)
        order = sorted(range(len(dataset)),key=lambda i:dataset.lengths[i])[rank::world]
        if config.get('sequence_packing'):
            packs = best_fit(order,dataset.lengths,config['max_sequence_length'])
            loader = make_loader(dataset,inputs.PackedDecisionCollator(tokenizer,config['max_sequence_length']),
                                 config['workers'],batch_sampler=packs)
        else:
            loader = make_loader(Subset(dataset,order),inputs.DecisionCollator(tokenizer,config.get('max_sequence_length')),
                                 config['workers'],batch_size=config['eval_batch_size'])
        torch.cuda.synchronize()
        start = time.monotonic()
        results = predict(model,loader,device)
        if hasattr(model, 'last_video_cache_stats'):
            save_json(model.last_video_cache_stats, output/f'{name}_video_cache_rank{rank}.json')
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.monotonic()-start,device=device)
        atomic_save(results,output/f'{name}_rank{rank}.pt')
        if world > 1:
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            dist.barrier()
        if rank == 0:
            raw = [s for r in range(world) for s in torch.load(output/f'{name}_rank{r}.pt',weights_only=True)]
            by_id = {r['id']:r for r in raw}
            assert len(by_id) == len(raw) == len(dataset)
            assert set(by_id) == {r['id'] for r in dataset.rows}
            scores = [by_id[r['id']] for r in dataset.rows]
            timing[name] = dict(seconds=float(elapsed),queries_per_second=len(dataset)/float(elapsed),
                world_size=world,per_gpu_batch_size=config['eval_batch_size'],sample_fps=config['eval_fps'],
                formulation=FORMULATION,sequence_packing=bool(config.get('sequence_packing')),
                max_sequence_length=config.get('max_sequence_length'),scope='native feature read + decision inputs + language/shared-head forward')
            if hasattr(model, 'last_video_cache_stats'):
                timing[name]['video_cache'] = [json.loads((output/f'{name}_video_cache_rank{r}.json').read_text())
                                               for r in range(world)]
            atomic_save(dict(scores=scores,timing=timing[name],checkpoint_id=checkpoint_id,
                visual_protocol=VISUAL_PROTOCOL,sample_fps=config['eval_fps'],query_placement='last',
                attention_mode=config['attention_mode'],formulation=FORMULATION,
                decision_metadata=model.decision_metadata,max_sequence_length=config.get('max_sequence_length'),
                feature_variant_root=config.get('feature_variant_root'),manifest_sha256=sha256(config['benchmarks'][name])),
                output/f'{name}_scores.pt')
            all_rows[name],all_scores[name] = dataset.rows,scores
            print(json.dumps(dict(event='benchmark_forward',dataset=name,**timing[name])),flush=True)
        del loader,dataset
        if world > 1:
            dist.barrier()
    decoder,summary = None,None
    if rank == 0:
        decoder = select_decoder(all_rows,all_scores,config['decoder_thresholds'],config['decoder_smoothing'])
        decoder.update(checkpoint_id=checkpoint_id,visual_protocol=VISUAL_PROTOCOL,sample_fps=config['eval_fps'],
            query_placement='last',attention_mode=config['attention_mode'],formulation=FORMULATION,
            decision_metadata=model.decision_metadata)
        summary = export(all_rows,all_scores,decoder,output,timing)
        save_json(model.decision_metadata,output/'decision_metadata.json')
    result = [decoder,summary]
    if world > 1:
        dist.broadcast_object_list(result)
    return tuple(result)
