"""Score the seven official MLVU-Dev multiple-choice tasks, including macro M-Avg."""
from collections import Counter
import math
from pathlib import Path

QUESTIONS = 2174
VIDEOS = 1122
TASK_COUNTS = dict(plotQA=539, needle=355, ego=352, count=206, order=259,
                   anomaly_reco=200, topic_reasoning=263)


def formatted_options(row):
    return [f'{chr(65+i)}. {text}' for i,text in enumerate(row['candidates'])]


def validate_reference(reference):
    if (len(reference)!=QUESTIONS or len({r['official_id'] for r in reference})!=QUESTIONS
            or len({r['video'] for r in reference})!=VIDEOS
            or Counter(r['task_type'] for r in reference)!=TASK_COUNTS):
        raise ValueError('Expected all 2174 MLVU-Dev MCQs, 1122 videos, and seven official tasks')
    for row in reference:
        if (not row['question'].strip() or len(row['candidates'])!=4
                or row['answer'] not in row['candidates']):
            raise ValueError('Expected a complete question, four original options, and public answer')
    return {r['official_id']:r for r in reference}


def score_mlvu(predictions, rows, reference):
    official = validate_reference(reference)
    by_id = {p['id']:p for p in predictions}
    if (len(rows)!=QUESTIONS or len(predictions)!=QUESTIONS or len(by_id)!=QUESTIONS
            or len({r['id'] for r in rows})!=QUESTIONS
            or {r['question_id'] for r in rows}!=set(official)
            or set(by_id)!={r['id'] for r in rows}):
        raise ValueError('Missing, duplicate, or unknown MLVU question IDs')
    totals, correct = Counter(), Counter()
    answers, scored = {}, []
    for row in rows:
        target = official[row['question_id']]
        answer_index = target['candidates'].index(target['answer'])
        if (row['id']!='mlvu/'+target['official_id'] or row['query']!=target['question']
                or row['options']!=formatted_options(target) or row['answer_index']!=answer_index
                or row['video_key']!='mlvu/'+Path(target['video']).stem
                or row['official_duration']!=target['duration'] or row['task_type']!=target['task_type']):
            raise ValueError(f'Question differs from the pinned official release: {row["id"]}')
        prediction = by_id[row['id']]
        choice = prediction['answer_index']
        probabilities, logits = prediction['probabilities'], prediction['logits']
        if (prediction.get('task')!='choice' or prediction['options']!=formatted_options(target)
                or type(choice) is not int or not 0<=choice<4
                or len(probabilities)!=4 or len(logits)!=4
                or not all(math.isfinite(x) for x in [*probabilities,*logits])
                or any(x<0 or x>1 for x in probabilities)
                or not math.isclose(sum(probabilities),1.,abs_tol=1e-4)
                or choice!=max(range(4),key=lambda i:probabilities[i])):
            raise ValueError(f'Invalid MLVU prediction: {row["id"]}')
        hit = choice==answer_index
        task = target['task_type']
        totals[task]+=1;correct[task]+=hit
        answers[target['official_id']] = chr(65+choice)
        scored.append(dict(prediction,question_id=target['official_id'],task_type=task,
                           answer_letter=chr(65+choice),correct=hit))
    number_correct = sum(correct.values())
    by_task = {task:dict(n=totals[task],correct=correct[task],accuracy=correct[task]/totals[task])
               for task in TASK_COUNTS}
    m_avg = sum(row['accuracy'] for row in by_task.values())/len(TASK_COUNTS)
    metrics = dict(n=QUESTIONS,videos=VIDEOS,correct=number_correct,accuracy=number_correct/QUESTIONS,
        m_avg=m_avg,m_avg_percent=100*m_avg,primary_metric='m_avg',by_task=by_task,
        benchmark='MLVU',split='dev',tasks='multiple_choice_only',
        protocol='mlvu_official_dev_mcq_lmms_eval_v1',use_subtitles=False,use_audio=False,
        answer_letter_rule='first matching candidate, as in the official lmms-eval dataset',
        repeated_answer_candidates=sum(r['candidates'].count(r['answer'])>1 for r in reference))
    return metrics, answers, scored
