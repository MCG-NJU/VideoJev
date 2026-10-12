"""Strict, full-coverage LVBench multiple-choice evaluation."""

import json
import math
from collections import defaultdict
from pathlib import Path




def score_lvbench(predictions, rows, metadata):
    """Match official category accuracy after enforcing all 1549 UID answers."""
    if len(rows) != 1549 or len(predictions) != len(rows):
        raise ValueError('LVBench requires exactly 1549 reference questions and predictions')
    by_id = {prediction['id']: prediction for prediction in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row['id'] for row in rows}:
        raise ValueError('Missing, duplicated or unknown LVBench prediction IDs')
    official = [json.loads(line) for line in Path(metadata).read_text().splitlines() if line.strip()]
    answers = {str(qa['uid']): qa for video in official for qa in video['qa']}
    owners = {str(qa['uid']): video['key'] for video in official for qa in video['qa']}
    if len(answers) != 1549 or {row['uid'] for row in rows} != set(answers):
        raise ValueError('Evaluation manifest differs from the complete official LVBench inventory')
    totals = defaultdict(int)
    correct = defaultdict(int)
    answer_file = {}
    output_rows = []
    for row in rows:
        prediction = by_id[row['id']]
        uid = row['uid']
        options = row['options']
        choice = prediction['answer_index']
        from .qa_manifest import parse_choice
        query, original_options, answer = parse_choice(answers[uid]['question'], answers[uid]['answer'])
        if (row['query'] != query or options != original_options or row['answer_index'] != answer
                or row['video_key'] != 'lvbench/' + owners[uid]):
            raise ValueError(f'LVBench row differs from the official reference: {uid}')
        probabilities = prediction['probabilities']
        logits = prediction['logits']
        if (type(choice) is not int or not 0 <= choice < len(options)
                or prediction.get('task') != 'choice' or prediction['options'] != options
                or len(probabilities) != len(options) or len(logits) != len(options)
                or not all(math.isfinite(x) for x in [*probabilities, *logits])
                or any(x < 0 or x > 1 for x in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-4)
                or choice != max(range(len(options)), key=lambda i: probabilities[i])):
            raise ValueError(f'Invalid options or prediction for LVBench UID {uid}')
        letter = chr(65 + choice)
        answer_file[uid] = letter
        target = answers[uid]
        if letter == target['answer']:
            correct['acc'] += 1
        totals['acc'] += 1
        for category in target['question_type']:
            totals[category] += 1
            if letter == target['answer']:
                correct[category] += 1
        output_rows.append(dict(prediction, uid=uid, answer_letter=letter,
                                correct=letter == target['answer']))
    metrics = {key: correct[key] / count for key, count in totals.items()}
    return dict(n=1549, correct=correct['acc'], accuracy=metrics['acc'],
                accuracy_by_category=metrics, total_by_category=dict(totals)), answer_file, output_rows
