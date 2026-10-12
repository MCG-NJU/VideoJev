"""One native training-style forward per complete-query chunk of the same video."""
import numpy as np
import torch

from .data import to_device
from .multiturn_data import MultiturnCollator
from .multiturn_inputs import continuation_text, episode_inputs
from .timestamp_option_inputs import cached_inputs, timestamp_texts

PROTOCOL = 'greedy_video_query_chunks_single_forward_v1'


def query_chunks(processor, record, rows, context):
    if not rows or context < 64:
        raise ValueError('Expected a nonempty video and a valid context')
    if len({r['video_key'] for r in rows}) != 1 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Query chunks require unique questions of one video')
    stamps = timestamp_texts(processor, record) if any(r.get('task','grounding')=='grounding' for r in rows) else None
    first = [cached_inputs(processor, record, row, lengths_only=True) for row in rows]
    continuation = [len(processor.tokenizer.encode(continuation_text(processor,row,stamps),
                     add_special_tokens=False)) for row in rows]
    start, chunks = 0, []
    while start < len(rows):
        end, tokens = start+1, first[start]
        if (tokens+63)//64*64 > context:
            raise ValueError(f'Single intact query exceeds context; no question was skipped: {rows[start]["id"]}')
        while end < len(rows) and (tokens+continuation[end]+63)//64*64 <= context:
            tokens += continuation[end]
            end += 1
        chunks.append(dict(start=start, end=end, tokens=tokens))
        start = end
    return chunks


@torch.inference_mode()
def predict_video(model, processor, features, record, rows, context, device):
    chunks = query_chunks(processor,record,rows,context)
    collator = MultiturnCollator(processor.tokenizer,context)
    features = features.to(device)
    predictions = []
    bounds = record.get('score_bounds', record.get('audit',{}).get('score_bounds'))
    for chunk_index, chunk in enumerate(chunks):
        group = rows[chunk['start']:chunk['end']]
        inputs = episode_inputs(processor,record,group)
        if inputs['input_ids'].shape[1] != chunk['tokens']:
            raise ValueError('Native multi-query input differs from exact chunk length')
        inputs['visual_features'] = features
        labels = [torch.zeros(len(r['options']) if r.get('task')=='choice' else len(record['score_times'])) for r in group]
        item = dict(rows=[dict(r,sampled_duration=record['sampled_duration']) for r in group],
            inputs=inputs, times=np.asarray(record['score_times']), labels=labels,
            audit=dict(feature_cache_id=record['cache_id'], score_bounds=bounds))
        batch = collator([item])
        batch.pop('labels')
        # KEEP episode_lengths and turn_episode_indices. No answer/prediction text
        # is appended, and there is no persistent KV/DeltaNet state across chunks.
        out = model(**to_device(batch,device))
        logits, probabilities = out['logits'].float().cpu(), out['probabilities'].float().cpu()
        for i, row in enumerate(group):
            count = len(labels[i])
            if not torch.isfinite(logits[i,:count]).all() or not torch.isfinite(probabilities[i,:count]).all():
                raise ValueError(f'Nonfinite prediction: {row["id"]}')
            prediction = dict(id=row['id'], task=row.get('task','grounding'),
                logits=logits[i,:count].tolist(), probabilities=probabilities[i,:count].tolist(),
                audit=dict(protocol=PROTOCOL, video_key=row['video_key'],
                    chunk_index=chunk_index, chunks_per_video=len(chunks),
                    original_video_turn_index=chunk['start']+i, episode_tokens=chunk['tokens'],
                    max_sequence_length=context, forward_calls_per_chunk=1))
            if row.get('task') == 'choice':
                answer = int(probabilities[i,:count].argmax())
                prediction.update(answer_index=answer, answer=row['options'][answer], options=row['options'])
            else:
                prediction.update(times=record['score_times'], bounds=bounds,
                                  sampled_duration=record['sampled_duration'])
            predictions.append(prediction)
        del inputs, item, batch, out
    return dict(predictions=predictions, forward_calls=len(chunks), chunks=chunks)
