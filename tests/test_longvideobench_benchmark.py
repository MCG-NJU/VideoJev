import copy
import json
from pathlib import Path
import unittest

from timejev.longvideobench_benchmark import score_longvideobench, validate_reference
from timejev.qa_eval_settings import resolve_evaluation


class LongVideoBenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(resolve_evaluation('longvideobench')['official_questions']).is_file():
            raise unittest.SkipTest('Official benchmark annotations are not bundled with the code')
        evaluation=resolve_evaluation('longvideobench')
        cls.reference=json.loads(Path(evaluation['official_questions']).read_text())
        cls.rows=[json.loads(line) for line in Path(evaluation['manifest']).read_text().splitlines()]
        cls.predictions=[]
        for row in cls.rows:
            n=len(row['options']);choice=row['answer_index']
            probabilities=[float(i==choice) for i in range(n)]
            cls.predictions.append(dict(id=row['id'],task='choice',options=row['options'],
                answer_index=choice,probabilities=probabilities,logits=probabilities))

    def test_full_reference_and_original_candidate_counts(self):
        validate_reference(self.reference)
        metrics,answers,rows=score_longvideobench(self.predictions,self.rows,self.reference)
        self.assertEqual(metrics['n'],1337)
        self.assertEqual(metrics['videos'],753)
        self.assertEqual(metrics['correct'],1337)
        self.assertEqual(metrics['accuracy'],1)
        self.assertEqual(metrics['by_option_count']['3']['n'],1)
        self.assertEqual(metrics['by_option_count']['4']['n'],353)
        self.assertEqual(metrics['by_option_count']['5']['n'],983)
        self.assertEqual(sum(answer=='E' for answer in answers.values()),182)

    def test_accuracy_is_the_fraction_of_matching_official_indices(self):
        predictions=copy.deepcopy(self.predictions)
        for i,row in enumerate(predictions):
            if i%3:continue
            choice=(row['answer_index']+1)%len(row['options'])
            row['answer_index']=choice
            row['probabilities']=row['logits']=[float(j==choice) for j in range(len(row['options']))]
        metrics,_,_=score_longvideobench(predictions,self.rows,self.reference)
        self.assertEqual(metrics['correct'],1337-len(range(0,1337,3)))
        self.assertEqual(metrics['accuracy'],metrics['correct']/1337)
        for name in ('duration','level','question_category','topic_category','option_count'):
            self.assertEqual(sum(r['n'] for r in metrics['by_'+name].values()),1337)
            self.assertEqual(sum(r['correct'] for r in metrics['by_'+name].values()),metrics['correct'])

    def test_incomplete_duplicate_and_unknown_ids_rejected(self):
        for predictions in (self.predictions[:-1],self.predictions[:-1]+[self.predictions[0]]):
            with self.assertRaises(ValueError):score_longvideobench(predictions,self.rows,self.reference)
        predictions=copy.deepcopy(self.predictions);predictions[0]['id']='unknown'
        with self.assertRaises(ValueError):score_longvideobench(predictions,self.rows,self.reference)

    def test_changed_reference_inputs_rejected(self):
        for key,value in [('query','changed'),('options',['changed']*4),('answer_index',3),
                          ('video_key','wrong'),('duration',1)]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaises(ValueError):score_longvideobench(self.predictions,rows,self.reference)

    def test_invalid_numerics_or_candidate_padding_rejected(self):
        index=next(i for i,r in enumerate(self.rows) if len(r['options'])==3)
        for key,value in [('answer_index',3),('probabilities',[0,0,0,1]),
                          ('probabilities',[float('nan'),0,1]),('logits',[0,1])]:
            predictions=copy.deepcopy(self.predictions);predictions[index][key]=value
            with self.assertRaises(ValueError):score_longvideobench(predictions,self.rows,self.reference)

    def test_hidden_test_answers_are_not_treated_as_labels(self):
        reference=copy.deepcopy(self.reference);reference[0].pop('correct_choice')
        with self.assertRaises(ValueError):validate_reference(reference)

    def test_fps_defaults_and_explicit_alternative(self):
        self.assertEqual(resolve_evaluation('longvideobench')['eval_fps'],2)
        self.assertEqual(resolve_evaluation('longvideobench',4)['eval_fps'],4)


if __name__=='__main__':unittest.main()
