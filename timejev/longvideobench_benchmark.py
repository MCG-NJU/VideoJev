"""Strict scoring of the complete, publicly labeled LongVideoBench validation split."""
from collections import Counter
import math

QUESTIONS = 1337
VIDEOS = 753
DURATION_NAMES = {15: '8s-15s', 60: '15s-60s', 600: '180s-600s', 3600: '900s-3600s'}


def formatted_options(reference_row):
    # Both the official loader and the trained TimeJev prompts include these letters.
    return [f'{chr(65+i)}. {text}' for i,text in enumerate(reference_row['candidates'])]


def validate_reference(reference):
    if len(reference) != QUESTIONS or len({r['id'] for r in reference}) != QUESTIONS:
        raise ValueError('LongVideoBench validation requires 1337 unique official IDs')
    if len({r['video_id'] for r in reference}) != VIDEOS:
        raise ValueError('LongVideoBench validation requires 753 videos')
    if Counter(r['duration_group'] for r in reference) != {15:189, 60:172, 600:412, 3600:564}:
        raise ValueError('Official LongVideoBench duration inventory changed')
    if Counter(len(r['candidates']) for r in reference) != {3:1, 4:353, 5:983}:
        raise ValueError('Preserve all original 3/4/5-choice validation questions')
    for row in reference:
        if type(row.get('correct_choice')) is not int or not 0 <= row['correct_choice'] < len(row['candidates']):
            raise ValueError('Validation requires public ground truth; hidden test answers cannot be scored')
    return {row['id']:row for row in reference}


def score_longvideobench(predictions, rows, reference):
    official = validate_reference(reference)
    by_id = {p['id']:p for p in predictions}
    if (len(rows) != QUESTIONS or len(predictions) != QUESTIONS or len(by_id) != QUESTIONS
            or len({r['id'] for r in rows}) != QUESTIONS
            or {r['question_id'] for r in rows} != set(official)
            or set(by_id) != {r['id'] for r in rows}):
        raise ValueError('Missing, duplicate, or unknown LongVideoBench IDs')
    totals = {name:Counter() for name in ('duration_group','level','question_category','topic_category','option_count')}
    correct = {name:Counter() for name in totals}
    answers, scored = {}, []
    for row in rows:
        target = official[row['question_id']]
        if (row['id'] != 'longvideobench/'+target['id'] or row['query'] != target['question']
                or row['options'] != formatted_options(target) or row['answer_index'] != target['correct_choice']
                or row['video_key'] != 'longvideobench/'+target['video_id']
                or row['duration'] != target['duration']):
            raise ValueError(f'Question differs from the official reference: {row["id"]}')
        prediction = by_id[row['id']]
        n = len(target['candidates'])
        choice, probabilities, logits = prediction['answer_index'], prediction['probabilities'], prediction.get('logits', prediction['probabilities'])
        if (prediction.get('task') != 'choice' or prediction['options'] != formatted_options(target)
                or type(choice) is not int or not 0 <= choice < n
                or len(probabilities) != n or len(logits) != n
                or not all(math.isfinite(x) for x in [*probabilities,*logits])
                or any(x < 0 or x > 1 for x in probabilities)
                or not math.isclose(sum(probabilities),1.,abs_tol=1e-4)
                or choice != max(range(n),key=lambda i:probabilities[i])):
            raise ValueError(f'Invalid LongVideoBench prediction: {row["id"]}')
        hit = choice == target['correct_choice']
        answers[target['id']] = chr(65+choice)
        scored.append(dict(prediction,question_id=target['id'],answer_letter=chr(65+choice),correct=hit))
        for name in totals:
            group = n if name == 'option_count' else target[name]
            totals[name][group] += 1
            correct[name][group] += hit
    number_correct = sum(row['correct'] for row in scored)
    metrics = dict(n=QUESTIONS,videos=VIDEOS,correct=number_correct,accuracy=number_correct/QUESTIONS,
                   benchmark='LongVideoBench',split='validation',protocol='longvideobench_official_validation_choice_v1',
                   setting='without_subtitles',use_subtitles=False,use_audio=False)
    for name in totals:
        key = 'duration' if name == 'duration_group' else name
        metrics['by_'+key] = {(DURATION_NAMES[group] if name=='duration_group' else str(group)):
            dict(n=count,correct=correct[name][group],accuracy=correct[name][group]/count)
            for group,count in sorted(totals[name].items())}
    return metrics, answers, scored
