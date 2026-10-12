import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from timejev.videomme_benchmark import score_videomme
from timejev.qa_eval_settings import resolve_evaluation

SUITE=Path(resolve_evaluation('videomme')['official_questions']).parent


class VideoMMETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(resolve_evaluation('videomme')['official_questions']).is_file():
            raise unittest.SkipTest('Official benchmark annotations are not bundled with the code')
        cls.reference=json.loads((Path(resolve_evaluation('videomme')['official_questions'])).read_text())
        cls.rows=[];cls.predictions=[]
        for i,r in enumerate(cls.reference):
            cls.rows.append(dict(id='videomme/'+r['question_id'],question_id=r['question_id'],
                query=r['question'],options=r['options'],answer_index=ord(r['answer'])-65,
                video_key='videomme/'+r['videoID']))
            # Deliberately non-perfect, duration-varying choices exercise all totals.
            answer=(ord(r['answer'])-65+(i%5==0))%4
            probabilities=[float(j==answer) for j in range(4)]
            cls.predictions.append(dict(id=cls.rows[-1]['id'],task='choice',answer_index=answer,
                probabilities=probabilities,logits=probabilities,options=r['options']))

    def test_official_scoring_agrees_for_full_dataset_and_durations(self):
        metrics,answers,_,official=score_videomme(self.predictions,self.rows,self.reference)
        self.assertEqual(metrics['n'],2700)
        self.assertEqual(metrics['correct'],2160)
        self.assertEqual(len(answers),2700)
        self.assertEqual(len(official),900)
        spec=importlib.util.spec_from_file_location('official_scorer',SUITE/'eval_your_results.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.json';path.write_text(json.dumps(official))
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                module.eval_your_results(str(path),video_types='short,medium,long',
                    return_categories_accuracy=False)
        values=re.findall(r'Overall:\s*([0-9.]+)%',stream.getvalue())
        expected=[metrics['by_duration'][key]['accuracy']*100 for key in ('short','medium','long')]+[metrics['accuracy']*100]
        self.assertEqual(values,[f'{x:.1f}' for x in expected])

    def test_reject_missing_and_duplicate_predictions(self):
        with self.assertRaises(ValueError):score_videomme(self.predictions[:-1],self.rows,self.reference)
        predictions=self.predictions[:-1]+[self.predictions[0]]
        with self.assertRaises(ValueError):score_videomme(predictions,self.rows,self.reference)

    def test_reject_changed_options_and_answer(self):
        rows=copy.deepcopy(self.rows);rows[0]['options']=list(reversed(rows[0]['options']))
        with self.assertRaises(ValueError):score_videomme(self.predictions,rows,self.reference)
        rows=copy.deepcopy(self.rows);rows[0]['answer_index']=(rows[0]['answer_index']+1)%4
        with self.assertRaises(ValueError):score_videomme(self.predictions,rows,self.reference)

    def test_reject_invalid_choice_and_probabilities(self):
        for key,value in [('answer_index',4),('probabilities',[float('nan'),0,0,1]),('logits',[0,1,2])]:
            predictions=copy.deepcopy(self.predictions);predictions[0][key]=value
            with self.assertRaises(ValueError):score_videomme(predictions,self.rows,self.reference)

    def test_reject_changed_video_inventory(self):
        rows=copy.deepcopy(self.rows);rows[0]['video_key']='videomme/wrong'
        with self.assertRaises(ValueError):score_videomme(self.predictions,rows,self.reference)


if __name__=='__main__':unittest.main()
