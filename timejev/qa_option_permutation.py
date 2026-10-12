"""Reproducible, answer-independent choice permutations for prepared QA manifests.

One permutation per question is sufficient for the one-epoch experiments. Source
records with explicit option references stay intact; neither their turns nor
their choices are reordered. Evaluation never calls this transformation.
"""
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL = 'qa_option_permutation_sha256_v1'
LABEL = re.compile(r'^([A-Z])\. (.*)$', re.DOTALL)
REFERENCE_PATTERNS = {
    'relative_options': re.compile(
        r'\b(?:all|none|both|any|neither|each|either|one|two|three|four|five|some)\s+'
        r'(?:(?:of|are)\s+)?(?:the\s+)?(?:above|below|other|previous|preceding|following)\b', re.I),
    'named_option': re.compile(
        r'\b(?:option|choice|answer|statement|alternative)s?\s*[\(\[\:\"\x27]*[A-E]\b', re.I),
    'letter_combination': re.compile(r'\b[A-E]\s*(?:[,/&+\-→]|and\b|or\b|then\b|to\b)\s*[A-E]\b'),
    'bracketed_letter': re.compile(r'\([A-E]\)|\[[A-E]\]|[\"\x27“‘][A-E][\"\x27”’]'),
    'letter_list': re.compile(r'^\s*[A-E][.:)]\s', re.M),
    'ordinal_option': re.compile(r'\b(?:first|second|third|fourth|fifth|last|former|latter)\s+(?:\w+\s+)?(?:options?|choices?|answers?)\b', re.I),
    'prior_turn_reference': re.compile(r'\b(?:previous|preceding|earlier|above|last)\s+(?:questions?|answers?|options?|choices?)\b', re.I),
    'numbered_option': re.compile(r'\b(?:options?|choices?|answers?|statements?)\s+[1-5]\b', re.I),
    'deictic_options': re.compile(r'\b(?:all|both|neither|none)\s+(?:of\s+)?(?:these|those)\b', re.I),
}


def option_bodies(row):
    if row.get('task') != 'choice' or not 2 <= len(row['options']) <= 26:
        raise ValueError('Permutation requires labelled single-choice QA')
    bodies = []
    for i, option in enumerate(row['options']):
        match = LABEL.fullmatch(option)
        if match is None or match[1] != chr(65+i) or not match[2].strip():
            raise ValueError(f'Malformed choice labels: {row["id"]}')
        bodies.append(match[2])
    answer = row['answer_index']
    if type(answer) is not int or not 0 <= answer < len(bodies):
        raise ValueError('Invalid single-choice answer')
    return bodies


def reference_reasons(row):
    bodies = option_bodies(row)
    text = '\n'.join([row['query'], *bodies])
    reasons = [name for name, pattern in REFERENCE_PATTERNS.items() if pattern.search(text)]
    if any(re.fullmatch(r'[A-E]', body.strip()) for body in bodies):
        reasons.append('bare_letter_option')
    return reasons


def record_key(row):
    return (row.get('source_file', row.get('source', '')), row.get('source_id', row['id']))


def permute_row(row, seed, *, epoch=0, preserve_reasons=()):
    bodies = option_bodies(row)
    reasons = sorted(set(reference_reasons(row)) | set(preserve_reasons))
    order = list(range(len(bodies)))
    if not reasons:
        # Never use labels, rank, Python's salted hash, or traversal order as RNG input.
        identity = json.dumps([PROTOCOL, int(seed), int(epoch), row['id']], separators=(',', ':'))
        rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), 'big'))
        rng.shuffle(order)
    out = copy.deepcopy(row)
    out['options'] = [f'{chr(65+i)}. {bodies[old]}' for i, old in enumerate(order)]
    out['answer_index'] = order.index(row['answer_index'])
    if bodies[row['answer_index']] != option_bodies(out)[out['answer_index']]:
        raise AssertionError('Permutation changed the correct answer text')
    audit = dict(id=row['id'], source=row.get('source'), new_to_old=order,
                 original_answer_index=row['answer_index'], answer_index=out['answer_index'],
                 preserved=bool(reasons), reasons=reasons,
                 changed=order != list(range(len(order))))
    return out, audit


def prepare_manifest(source, output, *, seed, epoch=0):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    blocked = defaultdict(set)
    for row in rows:
        blocked[record_key(row)].update(reference_reasons(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_suffix('.permutations.jsonl')
    before, after, reasons, sizes = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    preserved = changed = 0
    with output.with_suffix('.tmp').open('w') as dest, audit_path.with_suffix('.tmp').open('w') as audit:
        for row in rows:
            new, trace = permute_row(row, seed, epoch=epoch, preserve_reasons=blocked[record_key(row)])
            dest.write(json.dumps(new, ensure_ascii=False, separators=(',', ':'))+'\n')
            audit.write(json.dumps(trace, ensure_ascii=False, separators=(',', ':'))+'\n')
            key = f'{row.get("source")}/{len(row["options"])}'
            before[key][row['answer_index']] += 1
            after[key][new['answer_index']] += 1
            sizes[len(row['options'])] += 1
            preserved += trace['preserved']; changed += trace['changed']
            reasons.update(trace['reasons'])
    output.with_suffix('.tmp').replace(output)
    audit_path.with_suffix('.tmp').replace(audit_path)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(protocol=PROTOCOL, seed=seed, epoch=epoch, questions=len(rows),
                  original_records=len(blocked), preserved_records=sum(bool(v) for v in blocked.values()),
                  eligible_questions=len(rows)-preserved, preserved_questions=preserved,
                  changed_questions=changed, preserved_reason_counts=dict(reasons),
                  candidate_counts=dict(sizes), labels_before=dict(before), labels_after=dict(after),
                  source=str(source), source_sha256=digest(source), manifest=str(output),
                  manifest_sha256=digest(output), audit=str(audit_path), audit_sha256=digest(audit_path),
                  transform_sha256=digest(__file__),
                  correct_answer_text_preserved=True, record_and_question_order_preserved=True,
                  permutation_independent_of_correct_answer=True, evaluation_permuted=False)
    report = output.with_suffix('.permutation_report.json')
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
