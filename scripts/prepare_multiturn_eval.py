"""Freeze the completed multi-query experiment's exact features and question order."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROTOCOL = 'greedy_video_query_chunks_single_forward_v1'
COUNTS = dict(charades=3363, activitynet=4500, qvhighlights=1541,
              lvbench=1549, videomme=2700, longvideobench=1337, mlvu=2174)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_plan(reference, destination):
    """Reuse input identities, never the original checkpoint or its predictions."""
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    inventory = read(reference/'identity.json')['files']
    selected = {p: h for p, h in inventory.items()
                if p in ('plan.json', 'decoder.json') or p.startswith('inputs/')}
    for name, expected in selected.items():
        if sha(reference/name) != expected:
            raise ValueError(f'Multi-query reference input changed: {name}')
    old = read(reference/'plan.json')
    expected_scopes = {f'{c}/{n}' for n in COUNTS for c in
                       ((147456,) if n in ('charades','activitynet','qvhighlights') else (196608,262144))}
    if set(old['scopes']) != expected_scopes:
        raise ValueError('Multi-query reference must contain all eleven evaluations')
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied, scopes = {}, {}

    def copy_input(path):
        path = Path(path)
        relative = str(path.relative_to(reference))
        if relative not in selected:
            raise ValueError(f'Unbound reference file: {path}')
        target = destination.parent/'multiturn_inputs'/path.name
        shutil.copy2(path, target)
        name = str(target.relative_to(destination.parent))
        copied[name] = selected[relative]
        return name

    (destination.parent/'multiturn_inputs').mkdir(exist_ok=True)
    for scope, spec in old['scopes'].items():
        scopes[scope] = dict(spec, manifest=copy_input(spec['manifest']),
            reference={k:copy_input(v) for k,v in spec.get('reference',{}).items()})
        rows = [json.loads(line) for line in (destination.parent/scopes[scope]['manifest']).read_text().splitlines()]
        jobs = [j for j in old['jobs'] if j['scope'] == scope]
        indices = [i for job in jobs for i in job['query_indices']]
        if (len(rows) != COUNTS[spec['name']] or len({r['id'] for r in rows}) != len(rows)
                or sorted(indices) != list(range(len(rows)))
                or len({j['video_key'] for j in jobs}) != len(jobs)):
            raise ValueError(f'Incomplete/duplicate multi-query reference coverage: {scope}')
        for job in jobs:
            if (job['query_indices'] != sorted(job['query_indices'])
                    or any(rows[i]['video_key'] != job['video_key'] for i in job['query_indices'])
                    or job['context'] != spec['context'] or job['task'] != spec['task']):
                raise ValueError(f'Question ordering/video identity changed: {scope}')
    decoder = read(reference/'decoder.json')
    cache = Path(old['cache'])
    value = dict(protocol=PROTOCOL, base=read(cache/'model.json')['base'], scopes=scopes, jobs=old['jobs'],
        files=copied, cache=str(cache), cache_model_sha256=sha(cache/'model.json'),
        cache_completion_sha256=sha(cache/'completion.json'),
        decoder={k:decoder[k] for k in ('threshold','smooth','postprocess')},
        reference=str(reference), reference_plan_sha256=selected['plan.json'],
        decoder_source_sha256=selected['decoder.json'],
        policy='Manifest-order greedy complete-query chunks; repeat unchanged video prefix; no answer history; no skips')
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    return {str(reference/p):h for p,h in selected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Input plan bundle with identity.json, plan.json, decoder.json and inputs/')
    parser.add_argument('--out', required=True, help='Destination plan JSON')
    args = parser.parse_args()
    freeze_plan(args.reference, args.out)


if __name__ == '__main__':
    main()
