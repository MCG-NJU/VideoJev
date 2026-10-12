"""Explicit QA evaluation settings, independent of historical checkpoint defaults."""
import json
from .config import read_config
from pathlib import Path

DEFAULT_FPS = 2.0
FPS_CHOICES = (2.0, 4.0)
QA_CONTEXT = 262144


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
    result = dict(config, qa_task='choice', max_sequence_length=QA_CONTEXT,
                  feature_root=evaluation['feature_root'], feature_variant_root=None,
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


def write_qa_results(metrics, directory, *, predictions_folder='context_262144'):
    """Report the four QA benchmarks at the fixed 256K context."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    lines = ['# VideoJev QA results', '', '| Benchmark | Score (%) | Questions |', '|---|---:|---:|']
    for name in ('lvbench', 'videomme', 'longvideobench', 'mlvu'):
        result = metrics[name]
        if result.get('context', result.get('max_sequence_length', QA_CONTEXT)) != QA_CONTEXT:
            raise ValueError('QA results must use the fixed 262144-token context')
        metric = 'm_avg' if name == 'mlvu' else 'accuracy'
        reports[name] = dict(context=QA_CONTEXT, metric=metric, metrics=result,
                             predictions_directory=predictions_folder+'/'+name)
        lines.append(f"| {name} | {100*result[metric]:.4f} | {result['n']} |")
    (directory/'qa_results.json').write_text(json.dumps(reports, indent=2)+'\n')
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return [directory/'qa_results.json', directory/'RESULTS.md']
