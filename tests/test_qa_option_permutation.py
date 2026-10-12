import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from timejev.decision_inputs import suffix, CHOICE, GROUNDING
from timejev.decision_model import typed_outputs
from timejev.qa_option_permutation import permute_row, prepare_manifest, option_bodies


def row(count=4, identity='item/0'):
    return dict(id=identity, task='choice', query='What object is carried?',
                options=[f'{chr(65+i)}. object {i}' for i in range(count)], answer_index=1,
                video_key='v', video_path='/v.mp4', duration=8., source='test',
                source_id=identity.split('/')[0], turn_index=int(identity.split('/')[-1]))


@pytest.mark.parametrize('count', [2, 4, 5])
def test_permutation_keeps_semantics_and_does_not_depend_on_answers(count):
    original = row(count)
    saved = copy.deepcopy(original)
    first, audit = permute_row(original, 20260924)
    assert original == saved
    assert sorted(audit['new_to_old']) == list(range(count))
    assert option_bodies(first)[first['answer_index']] == option_bodies(original)[1]
    assert permute_row(original, 20260924) == (first, audit)
    for answer in range(count):
        changed, trace = permute_row(dict(original, answer_index=answer), 20260924)
        assert trace['new_to_old'] == audit['new_to_old']
        assert suffix(changed) == suffix(first), 'Labels must not alter inputs'
        assert option_bodies(changed)[changed['answer_index']] == option_bodies(original)[answer]


@pytest.mark.parametrize('text', ['All of the above', 'Both A and B', 'Option C is correct',
    'The first two options', '(B)', 'None of the preceding', 'A, C, B, D'])
def test_option_references_are_preserved(text):
    original = row(); original['options'][2] = 'C. '+text
    value, audit = permute_row(original, 23)
    assert value == original and audit['preserved']


def test_preparation_protects_whole_record_and_preserves_order(tmp_path):
    rows = [row(identity='record/0'), row(identity='record/1'), row(identity='other/0')]
    rows[1]['query'] = 'Which of the previous options fits?'
    source = tmp_path/'source.jsonl'; dest = tmp_path/'new.jsonl'
    text = ''.join(json.dumps(x)+'\n' for x in rows);source.write_text(text)
    report = prepare_manifest(source, dest, seed=31)
    output = [json.loads(x) for x in dest.read_text().splitlines()]
    assert output[:2] == rows[:2]
    assert [r['id'] for r in output] == [r['id'] for r in rows]
    assert source.read_text() == text
    assert report['preserved_records'] == 1 and report['preserved_questions'] == 2
    assert report['questions'] == 3 and not report['evaluation_permuted']
    with pytest.raises(FileExistsError): prepare_manifest(source, dest, seed=31)


def test_biased_answers_become_approximately_uniform_without_using_labels():
    for count in [4, 5]:
        histogram = Counter()
        for i in range(2000):
            x = row(count, f'q{i}/0'); x['answer_index'] = 0
            shuffled, _ = permute_row(x, 20260924)
            histogram[shuffled['answer_index']] += 1
        assert set(histogram) == set(range(count))
        assert all(abs(v/2000-1/count) < .04 for v in histogram.values())


