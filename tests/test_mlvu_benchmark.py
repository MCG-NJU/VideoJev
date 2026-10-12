import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from timejev.mlvu_benchmark import score_mlvu, validate_reference, TASK_COUNTS
from timejev.qa_eval_settings import resolve_evaluation


class MLVUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(resolve_evaluation('mlvu')['official_questions']).is_file():
            raise unittest.SkipTest('Official benchmark annotations are not bundled with the code')
        evaluation=resolve_evaluation('mlvu')
        cls.reference=json.loads(Path(evaluation['official_questions']).read_text())
        cls.rows=[json.loads(line) for line in Path(evaluation['manifest']).read_text().splitlines()]
        cls.predictions=[]
        for row in cls.rows:
            choice=row['answer_index'];probabilities=[float(i==choice) for i in range(4)]
            cls.predictions.append(dict(id=row['id'],task='choice',options=row['options'],
                answer_index=choice,probabilities=probabilities,logits=probabilities))
        source=Path(evaluation['official_questions']).parent/'lmms_utils.py'
        function=next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='mlvu_aggregate_results_dev')
        namespace={'eval_logger':SimpleNamespace(info=lambda *_:None)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        cls.official_aggregate=staticmethod(namespace['mlvu_aggregate_results_dev'])

    @staticmethod
    def select(prediction,index):
        prediction['answer_index']=index
        prediction['probabilities']=prediction['logits']=[float(i==index) for i in range(4)]

    def test_complete_seven_task_inventory(self):
        validate_reference(self.reference)
        metrics,answers,_=score_mlvu(self.predictions,self.rows,self.reference)
        self.assertEqual((metrics['n'],metrics['videos'],metrics['correct']),(2174,1122,2174))
        self.assertEqual(metrics['m_avg'],1)
        self.assertEqual(len(answers),2174)
        self.assertEqual({k:v['n'] for k,v in metrics['by_task'].items()},TASK_COUNTS)

    def test_macro_average_matches_official_implementation_not_micro_average(self):
        predictions=copy.deepcopy(self.predictions)
        official_results=[]
        for row,prediction in zip(self.rows,predictions):
            if row['task_type']!='plotQA':self.select(prediction,(row['answer_index']+1)%4)
            official_results.append(dict(task_type=row['task_type'],
                pred_answer=chr(65+prediction['answer_index']),answer=chr(65+row['answer_index'])))
        metrics,_,_=score_mlvu(predictions,self.rows,self.reference)
        self.assertAlmostEqual(metrics['m_avg'],1/7)
        self.assertAlmostEqual(metrics['accuracy'],539/2174)
        self.assertAlmostEqual(metrics['m_avg_percent'],self.official_aggregate(official_results))

    def test_duplicate_candidate_text_preserves_published_answer_letters(self):
        predictions=copy.deepcopy(self.predictions)
        duplicates=0
        for i,ref in enumerate(self.reference):
            matches=[j for j,text in enumerate(ref['candidates']) if text==ref['answer']]
            if len(matches)>1:
                duplicates+=1
                self.assertEqual(self.rows[i]['answer_index'],matches[0])
                self.select(predictions[i],matches[-1])
        self.assertEqual(duplicates,5)
        metrics,_,_=score_mlvu(predictions,self.rows,self.reference)
        self.assertEqual(metrics['correct'],2174-5)

    def test_missing_duplicate_and_unknown_ids_rejected(self):
        for predictions in (self.predictions[:-1],self.predictions[:-1]+[self.predictions[0]]):
            with self.assertRaises(ValueError):score_mlvu(predictions,self.rows,self.reference)
        predictions=copy.deepcopy(self.predictions);predictions[0]['id']='unknown'
        with self.assertRaises(ValueError):score_mlvu(predictions,self.rows,self.reference)

    def test_changed_official_inputs_rejected(self):
        for key,value in [('query','changed'),('options',['changed']*4),('answer_index',3),
                          ('video_key','wrong'),('official_duration',1),('task_type','order')]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaises(ValueError):score_mlvu(self.predictions,rows,self.reference)

    def test_invalid_predictions_rejected(self):
        for key,value in [('answer_index',4),('probabilities',[0,0,1]),
                          ('probabilities',[float('nan'),0,0,1]),('logits',[0,1])]:
            predictions=copy.deepcopy(self.predictions);predictions[0][key]=value
            with self.assertRaises(ValueError):score_mlvu(predictions,self.rows,self.reference)

    def test_fps_defaults_and_explicit_alternative(self):
        self.assertEqual(resolve_evaluation('mlvu')['eval_fps'],2)
        self.assertEqual(resolve_evaluation('mlvu',4)['eval_fps'],4)


if __name__=='__main__':unittest.main()
