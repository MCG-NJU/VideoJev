"""Full-inventory Video-MME scoring with the original choice order."""
from collections import Counter, defaultdict
import math


def validate_reference(reference):
    if len(reference) != 2700 or len({r['question_id'] for r in reference}) != 2700:
        raise ValueError('Video-MME requires 2700 unique official question IDs')
    counts = Counter(r['videoID'] for r in reference)
    if len(counts) != 900 or set(counts.values()) != {3}:
        raise ValueError('Video-MME requires 900 videos with three questions each')
    if Counter(r['duration'] for r in reference) != {'short':900, 'medium':900, 'long':900}:
        raise ValueError('Video-MME duration groups must each contain 900 questions')
    for row in reference:
        if len(row['options']) != 4 or row['answer'] not in 'ABCD' or len(row['answer']) != 1:
            raise ValueError('Video-MME requires four options and one A-D answer')
    return {row['question_id']: row for row in reference}


def score_videomme(predictions, rows, reference):
    official = validate_reference(reference)
    if len(rows) != 2700 or len(predictions) != 2700:
        raise ValueError('Missing Video-MME rows or predictions')
    by_id = {p['id']:p for p in predictions}
    if (len(by_id) != 2700 or len({r['id'] for r in rows}) != 2700
            or set(by_id) != {r['id'] for r in rows}
            or {r['question_id'] for r in rows} != set(official)):
        raise ValueError('Duplicated, missing, or unknown Video-MME IDs')
    totals = {key:Counter() for key in ('duration','domain','sub_category','task_type')}
    correct = {key:Counter() for key in totals}
    answers, output, videos = {}, [], {}
    for row in rows:
        target = official[row['question_id']]
        if (row['query'] != target['question'] or row['options'] != target['options']
                or row['answer_index'] != ord(target['answer'])-65
                or row['video_key'] != 'videomme/'+target['videoID']):
            raise ValueError(f'Evaluation row differs from official question: {row["id"]}')
        prediction = by_id[row['id']]
        choice = prediction['answer_index']
        probabilities = prediction['probabilities']
        logits = prediction['logits']
        if (type(choice) is not int or not 0 <= choice < 4
                or prediction.get('task') != 'choice' or prediction['options'] != target['options']
                or len(probabilities) != 4 or len(logits) != 4
                or not all(math.isfinite(x) for x in [*probabilities, *logits])
                or any(x < 0 or x > 1 for x in probabilities)
                or not math.isclose(sum(probabilities),1.,abs_tol=1e-4)
                or choice != max(range(4),key=lambda i:probabilities[i])):
            raise ValueError(f'Invalid choice prediction: {row["id"]}')
        letter = chr(65+choice)
        hit = letter == target['answer']
        answers[target['question_id']] = letter
        for key in totals:
            totals[key][target[key]] += 1
            correct[key][target[key]] += hit
        output.append(dict(prediction, question_id=target['question_id'],
                           answer_letter=letter, correct=hit))
        video = videos.setdefault(target['video_id'], {key:target[key] for key in
                                  ('video_id','duration','domain','sub_category')})
        video.setdefault('questions', []).append({**{key:target[key] for key in
            ('question_id','task_type','question','options','answer')}, 'response':letter})
    number_correct = sum(row['correct'] for row in output)
    metrics = dict(n=2700, videos=900, correct=number_correct, accuracy=number_correct/2700,
                   protocol='videomme_official_choice_v1', setting='without_subtitles',
                   use_subtitles=False, use_audio=False)
    for key in totals:
        metrics['by_'+key] = {group:dict(n=n, correct=correct[key][group],
                                        accuracy=correct[key][group]/n)
                              for group,n in sorted(totals[key].items())}
    return metrics, answers, output, [videos[key] for key in sorted(videos)]
