"""One native forward per query; tune one shared interval decoder on all three benchmarks."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from .data import Collator, GroundingDataset, read_jsonl
from .engine import atomic_save, make_loader, predict, save_json, sha256
from .temporal import decode, metrics, POSTPROCESS_VERSION
from .vision import VISUAL_PROTOCOL, load_processor

NAMES = ('charades', 'activitynet', 'qvhighlights')


def select_decoder(rows, scores, thresholds, smoothing, *, postprocess=POSTPROCESS_VERSION,
                   selection_postprocess='none'):
    """Select once, retaining both selection metrics and final output metrics.

    By default, select threshold/smoothing using continuous intervals, then
    round/repair/clamp the selected decoder's predictions (policy A). Separate
    arguments retain explicit legacy and alternative-policy comparisons.
    """
    candidates = []
    for threshold in thresholds:
        for smooth in smoothing:
            reports = {}
            for name in NAMES:
                predictions = [decode(s['logits'], s['times'], s['sampled_duration'], threshold, smooth,
                                      bounds=s['bounds'], postprocess=selection_postprocess)
                               for s in scores[name]]
                reports[name] = metrics(predictions, rows[name])
            candidates.append(dict(threshold=threshold, smooth=smooth, metrics=reports,
                                   selection_postprocess=selection_postprocess,
                                   mean_mIoU=float(np.mean([reports[n]['mIoU'] for n in NAMES]))))
    best = max(candidates, key=lambda x: (x['mean_mIoU'], -abs(x['threshold'] - .5), -x['smooth']))
    final_reports = best['metrics']
    if postprocess != selection_postprocess:
        final_reports = {name: metrics([
            decode(s['logits'], s['times'], s['sampled_duration'], best['threshold'], best['smooth'],
                   bounds=s['bounds'], postprocess=postprocess) for s in scores[name]], rows[name])
            for name in NAMES}
    result = dict(best)
    result.update(metrics=final_reports,
                  mean_mIoU=float(np.mean([final_reports[name]['mIoU'] for name in NAMES])),
                  selection_metrics=best['metrics'], selection_mean_mIoU=best['mean_mIoU'],
                  postprocess=postprocess, selection_postprocess=selection_postprocess,
                  selection_data=list(NAMES), objective='unweighted mean of three benchmark mIoU',
                  candidates=candidates, algorithm='maximum contiguous sum of threshold-shifted logits',
                  timestamp_semantics='native two-frame midpoint')
    if postprocess == POSTPROCESS_VERSION:
        result['clamp_duration'] = 'sampled_duration'
    return result


def export(rows, scores, decoder, output, timing):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Unlabelled saved decoders are historical, continuous outputs. Fresh
    # selection records the new protocol explicitly; reading old results never
    # silently changes their meaning.
    protocol = decoder.get('postprocess', 'none')
    final_decoder = dict(decoder, postprocess=protocol)
    summary = {}
    for name in NAMES:
        decoded = [decode(s['logits'], s['times'], s['sampled_duration'],
                          decoder['threshold'], decoder['smooth'], bounds=s['bounds'],
                          postprocess=protocol, return_raw=True) for s in scores[name]]
        predictions = [pair[0] for pair in decoded]
        report = metrics(predictions, rows[name])
        report.update(timing[name], coverage=f'{len(scores[name])}/{len(rows[name])}',
                      decoder={k: decoder[k] for k in ['threshold', 'smooth']}, postprocess=protocol)
        official, records = {}, []
        for row, score, (prediction, raw_prediction) in zip(rows[name], scores[name], decoded):
            assert len(row['spans']) == 1
            key = f'{Path(row["video_path"]).stem}>>>{row["query"]}>>>{row["spans"][0]}'
            if key in official:
                raise ValueError('Duplicate official key: ' + key)
            official[key] = dict(timestamps=[prediction], postprocess=protocol)
            logits = np.asarray(score['logits'])
            records.append(dict(**row, prediction=prediction, raw_prediction=raw_prediction,
                                postprocess=protocol, timestamps=score['times'],
                                score_bounds=score['bounds'],
                                logits=score['logits'], scores=(1 / (1 + np.exp(-np.clip(logits, -80, 80)))).tolist(),
                                sampled_duration=score['sampled_duration'], audit=score['audit']))
        save_json(report, output / f'{name}_metrics.json')
        save_json(official, output / f'{name}_official.json')
        (output / f'{name}_predictions.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        summary[name] = report
    save_json(summary, output / 'benchmark_summary.json')
    final_decoder['metrics'] = {
        name: {key: summary[name][key] for key in ('n', 'R1@0.3', 'R1@0.5', 'R1@0.7', 'mIoU')}
        for name in NAMES}
    final_decoder['mean_mIoU'] = float(np.mean([summary[name]['mIoU'] for name in NAMES]))
    if protocol == POSTPROCESS_VERSION:
        final_decoder['clamp_duration'] = 'sampled_duration'
    save_json(final_decoder, output / 'decoder.json')
    return summary


def load_scores(config, output, checkpoint_id):
    rows, scores, timing = {}, {}, {}
    for name in NAMES:
        rows[name] = read_jsonl(config['benchmarks'][name])
        blob = torch.load(Path(output) / f'{name}_scores.pt', weights_only=True)
        assert blob['visual_protocol'] == VISUAL_PROTOCOL
        assert blob['checkpoint_id'] == checkpoint_id
        assert blob['sample_fps'] == config['eval_fps']
        assert blob.get('query_placement', 'both') == config.get('query_placement', 'both')
        assert blob.get('attention_mode', 'causal') == config.get('attention_mode', 'causal')
        assert blob['manifest_sha256'] == sha256(config['benchmarks'][name])
        assert blob.get('max_sequence_length')==config.get('max_sequence_length')
        assert blob.get('feature_variant_root')==config.get('feature_variant_root')
        by_id = {r['id']: r for r in blob['scores']}
        assert len(by_id) == len(blob['scores']) == len(rows[name])
        assert set(by_id) == {r['id'] for r in rows[name]}
        scores[name] = [by_id[r['id']] for r in rows[name]]
        timing[name] = blob['timing']
    return rows, scores, timing


def evaluate_benchmarks(model, config, output, checkpoint_id, rank, world, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collator = Collator(load_processor(config['base']).tokenizer)
    all_rows, all_scores, timing = {}, {}, {}
    for name in NAMES:
        dataset = GroundingDataset(config['benchmarks'][name], config['base'],
                                   fps=config['eval_fps'], archive_root=config.get('eval_archive_root'),
                                   query_placement=config.get('query_placement', 'both'), feature_root=config.get('feature_root'),
                                   feature_variant_root=config.get('feature_variant_root'),
                                   fast_feature_inputs=config.get('sequence_packing',False),
                                   visual_token_budget=(config['max_sequence_length']*4//5 if config.get('max_sequence_length') else None))
        if config.get('sequence_packing'):
            from .packing import load_lengths,best_fit,PackedCollator
            dataset.lengths=load_lengths(config['packed_eval_lengths'][name],config['benchmarks'][name],config,dataset.rows)
        order = sorted(range(len(dataset)), key=lambda i: dataset.lengths[i])[rank::world]
        if config.get('sequence_packing'):
            packs=best_fit(order,dataset.lengths,config['max_sequence_length'])
            loader=make_loader(dataset,PackedCollator(load_processor(config['base']).tokenizer,config['max_sequence_length']),
                               config['workers'],batch_sampler=packs)
        else:
            loader = make_loader(Subset(dataset, order), collator, config['workers'],
                                 batch_size=config['eval_batch_size'])
        torch.cuda.synchronize()
        start = time.monotonic()
        results = predict(model, loader, device)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.monotonic() - start, device=device)
        atomic_save(results, output / f'{name}_rank{rank}.pt')
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.barrier()
        if rank == 0:
            raw = [s for r in range(world)
                   for s in torch.load(output / f'{name}_rank{r}.pt', weights_only=True)]
            by_id = {r['id']: r for r in raw}
            assert len(by_id) == len(raw) == len(dataset)
            assert set(by_id) == {r['id'] for r in dataset.rows}
            scores = [by_id[r['id']] for r in dataset.rows]
            timing[name] = dict(seconds=float(elapsed), queries_per_second=len(dataset) / float(elapsed),
                                world_size=world, per_gpu_batch_size=config['eval_batch_size'],
                                sample_fps=config['eval_fps'],
                                scope=('native merger feature read + official text assembly + H2D + language/head forward; frozen vision precomputed'
                                       if config.get('feature_root') else
                                       ('frame archive read + lossless RAM decompression' if config.get('eval_archive_root') else 'raw video decode')
                                       + ' + official preprocessing + H2D + full vision/text/head forward; no feature cache'))
            if config.get('sequence_packing'):
                timing[name].update(sequence_packing=True,max_sequence_length=config['max_sequence_length'],
                                    batch_unit='one packed sequence containing one or more isolated queries')
            blob = dict(scores=scores, timing=timing[name], checkpoint_id=checkpoint_id,
                        visual_protocol=VISUAL_PROTOCOL, sample_fps=config['eval_fps'],
                        query_placement=config.get('query_placement', 'both'), attention_mode=config.get('attention_mode', 'causal'),
                        max_sequence_length=config.get('max_sequence_length'),feature_variant_root=config.get('feature_variant_root'),
                        manifest_sha256=sha256(config['benchmarks'][name]))
            atomic_save(blob, output / f'{name}_scores.pt')
            all_rows[name], all_scores[name] = dataset.rows, scores
            print(json.dumps(dict(event='benchmark_forward', dataset=name, **timing[name])), flush=True)
        del loader, dataset
        if world > 1:
            dist.barrier()
    decoder, summary = None, None
    if rank == 0:
        decoder = select_decoder(all_rows, all_scores, config['decoder_thresholds'], config['decoder_smoothing'])
        decoder.update(checkpoint_id=checkpoint_id, visual_protocol=VISUAL_PROTOCOL, sample_fps=config['eval_fps'],
                       query_placement=config.get('query_placement', 'both'), attention_mode=config.get('attention_mode', 'causal'),
                       max_sequence_length=config.get('max_sequence_length'),sample_visual_fraction=config.get('sample_visual_fraction'))
        summary = export(all_rows, all_scores, decoder, output, timing)
    result = [decoder, summary]
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    return result
