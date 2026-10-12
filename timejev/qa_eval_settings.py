"""Explicit QA evaluation settings, independent of historical checkpoint defaults."""
import json
from .config import read_config
from pathlib import Path

DEFAULT_FPS = 2.0
FPS_CHOICES = (2.0, 4.0)


def add_evaluation_arguments(parser):
    parser.add_argument('--fps', type=float, choices=FPS_CHOICES, default=DEFAULT_FPS,
                        help='Requested evaluation fps (default: 2); full-video cap: 2048 frames')
    parser.add_argument('--evaluation-config', help='Optional matching manifest/cache/lengths JSON')


def resolve_evaluation(benchmark, fps=DEFAULT_FPS, path=None):
    if fps not in FPS_CHOICES:
        raise ValueError('QA evaluation supports only 2 or 4 fps')
    if path:
        evaluation = read_config(path)
    else:
        registry = Path(__file__).resolve().parents[1] / 'configs/qa_evaluation.json'
        evaluation = read_config(registry)[benchmark][str(int(fps))]
    if (evaluation.get('benchmark', benchmark) != benchmark
            or evaluation['eval_fps'] != fps or evaluation['max_frames'] != 2048):
        raise ValueError('Evaluation config disagrees with --fps or the 2048-frame cap')
    if evaluation.get('use_subtitles', False) or evaluation.get('use_audio', False):
        raise ValueError('QA evaluation currently uses video frames only')
    return dict(evaluation, benchmark=benchmark)


def apply_evaluation(config, evaluation):
    """Copy config; never alter training settings or the loaded checkpoint config."""
    if config.get('qa_task') != 'choice' and 'choice' not in config.get('training_tasks', []):
        raise ValueError('Checkpoint must support choice training')
    if evaluation.get('available') is False:
        raise FileNotFoundError('The requested evaluation cache is unavailable')
    if evaluation['eval_fps'] not in FPS_CHOICES or evaluation['max_frames'] != 2048:
        raise ValueError('Unsupported evaluation sampling settings')
    benchmark = evaluation['benchmark']
    result = dict(config, qa_task='choice', feature_root=evaluation['feature_root'], feature_variant_root=None,
                  eval_fps=evaluation['eval_fps'], max_frames=evaluation['max_frames'],
                  benchmarks={benchmark: evaluation['manifest']},
                  packed_eval_lengths={benchmark: evaluation['lengths']})
    if benchmark == 'lvbench':
        result['lvbench_metadata'] = evaluation['lvbench_metadata']
    return result


def validate_evaluation(config, evaluation):
    from .engine import sha256
    from .features import validate_cache
    spec = validate_cache(evaluation['feature_root'], [evaluation['manifest']], evaluation['eval_fps'])
    if (spec.get('requested_fps') != evaluation['eval_fps']
            or spec.get('max_frames') != evaluation['max_frames']):
        raise ValueError('Evaluation feature cache has a different sampling protocol')
    lengths = json.loads(Path(evaluation['lengths']).read_text())
    if (lengths.get('fps') != evaluation['eval_fps'] or lengths.get('cache_id') != spec['cache_id']
            or lengths.get('manifest_sha256') != sha256(evaluation['manifest'])
            or lengths.get('max_sequence_length') != config['max_sequence_length']):
        raise ValueError('Evaluation lengths do not match the fps/cache/manifest/context limit')
    return spec


def select_context_results(comparison):
    """Select one complete prediction set per benchmark; ties favor 192K."""
    contexts = (196608, 262144)
    names = ('lvbench', 'videomme', 'longvideobench', 'mlvu')
    result = {}
    for name in names:
        metric = 'm_avg' if name == 'mlvu' else 'accuracy'
        context = max(contexts, key=lambda c: (comparison[str(c)][name][metric], -c))
        result[name] = dict(context=context, metric=metric, metrics=comparison[str(context)][name])
    return result


def write_context_results(comparison, directory, *, folder_pattern='context_{context}'):
    directory = Path(directory)
    selected = select_context_results(comparison)
    lines = ['# VideoJev QA results', '', '| Benchmark | Score (%) | Questions |', '|---|---:|---:|']
    for name, value in selected.items():
        value['predictions_directory'] = folder_pattern.format(context=value['context'])+'/'+name
        metrics = value['metrics']
        lines.append(f"| {name} | {100*metrics[value['metric']]:.4f} | {metrics['n']} |")
    (directory/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return [directory/'selected.json', directory/'RESULTS.md']
