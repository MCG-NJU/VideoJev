"""Online RGB inputs with original 144K training and separate evaluation capacities."""
import json
from pathlib import Path

from .engine import sha256
from .rgb_data import RGBDataset, text_item as rgb_text_item

PROTOCOL = 'qwen36_mixed_rgb_context80_v1'
CAPACITY = 147456


def text_item(processor, rows, record, episode, capacity=CAPACITY):
    item = rgb_text_item(processor, rows, record, episode)
    item['audit'].update(rgb_protocol=PROTOCOL, feature_cache_id=PROTOCOL,
        max_sequence_length=capacity, visual_budget_fraction=.8,
        visual_token_budget=capacity*4//5, total_input_tokens=episode['tokens'])
    return item


class MixedRGBDataset(RGBDataset):
    def __getitem__(self, index):
        # Parent owns worker initialization, exact token checks and bounded RGB decoding.
        item = super().__getitem__(index)
        if 'audit' in item:
            item['audit'].update(rgb_protocol=PROTOCOL, feature_cache_id=PROTOCOL,
                max_sequence_length=self.config['max_sequence_length'], visual_budget_fraction=.8,
                visual_token_budget=self.config['max_sequence_length']*4//5)
        return item


def training_dataset(config):
    return MixedRGBDataset(config)


def collator(config, training=True):
    from .decision_inputs import load_processor
    from .multiturn_data import MultiturnCollator
    return MultiturnCollator(load_processor(config['base']).tokenizer, config['max_sequence_length'],
        pad_to_max_length=training, include_score_durations=training and bool(config['soft_iou_weight']))


def verify_data(config):
    from .decision_inputs import load_processor
    from .token_metadata import tokenizer_identity
    root = Path(config['rgb_ready']).parent
    ready = json.loads(Path(config['rgb_ready']).read_text())
    if not ready.get('complete'):
        raise ValueError('Mixed RGB preparation is incomplete')
    for name, value in ready['files'].items():
        if sha256(root/name) != value:
            raise ValueError(f'Mixed RGB data changed: {name}')
    identity = json.loads((root/'identity.json').read_text())
    if (identity['base'] != config['base'] or identity['capacity'] != config['max_sequence_length']
            or identity['tokenizer'] != tokenizer_identity(load_processor(config['base']), prompt_version=PROTOCOL)):
        raise ValueError('Prepared tokenizer/model/context differs from training')


def evaluation_schedule(spec, capacity, rank, world):
    """Balance total independent question work; keep every video on one rank."""
    loads, shards = [0]*world, [[] for _ in range(world)]
    for i in sorted(range(len(spec['videos'])),
                    key=lambda i: (-sum(q['tokens']**2 for q in spec['videos'][i]['questions']), i)):
        worker = min(range(world), key=lambda r: (loads[r], r))
        shards[worker].append(i)
        loads[worker] += sum(q['tokens']**2 for q in spec['videos'][i]['questions'])
    packs, current, length = [], [], 0
    for i in shards[rank]:
        for j, question in enumerate(spec['videos'][i]['questions']):
            if not 0 < question['tokens'] <= capacity:
                raise ValueError('Evaluation question exceeds its context')
            if current and length+question['tokens'] > capacity:
                packs.append(current)
                current, length = [], 0
            current.append((i, j))
            length += question['tokens']
    if current:
        packs.append(current)
    return shards[rank], packs
