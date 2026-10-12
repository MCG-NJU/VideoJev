"""Concatenate native timestamp datasets without conflating cache provenance."""
from bisect import bisect_right
import json
from pathlib import Path
from torch.utils.data import Dataset
from .data import read_jsonl
from .engine import sha256
from .timestamp_option_data import TimestampOptionDataset, load_lengths


def source_config(config, source):
    return dict(config, train=source['manifest'],
        train_feature_root=source['feature_root'],
        train_feature_variant_root=source.get('feature_variant_root'),
        train_media_shapes=source['media_shapes'], packed_train_lengths=source['lengths'])


def source_bindings(config):
    return [dict(source=source, manifest_sha256=sha256(source['manifest']),
        lengths_sha256=sha256(source['lengths']),
        cache_ids=[json.loads((Path(root)/'provenance.json').read_text())['cache_id']
            for root in (source['feature_root'], source.get('feature_variant_root')) if root])
        for source in config['training_sources']]


def provenance_paths(config):
    paths=[]
    for source in config['training_sources']:
        paths.extend([source['manifest'],source['media_shapes'],source['lengths']])
        for key in ('feature_root','feature_variant_root'):
            if source.get(key):
                paths.extend(str(Path(source[key])/name) for name in ('provenance.json','index.json'))
        if source.get('feature_variant_root'):
            paths.append(str(Path(source['feature_variant_root'])/'resolution_plan.json'))
    return paths


def validate_training_caches(config):
    from .features import validate_cache
    specs=[]
    for source in config['training_sources']:
        variant=source.get('feature_variant_root')
        specs.append(validate_cache(source['feature_root'],[source['manifest']],config['train_fps'],
            replacement_root=variant))
        if variant:
            specs.append(validate_cache(variant,[],config['train_fps'],require_complete=False))
    if not specs or len({spec['vision_sha256'] for spec in specs})!=1:
        raise ValueError('Mixed training caches must share exactly the same frozen vision weights')
    return specs


def load_mixture_lengths(config, rows):
    spec=json.loads(Path(config['packed_train_lengths']).read_text())
    if (spec.get('schema')!='timestamp_source_mixture_v1'
            or spec.get('manifest_sha256')!=sha256(config['train'])
            or spec.get('source_bindings')!=source_bindings(config)
            or spec.get('max_sequence_length')!=config['max_sequence_length']):
        raise ValueError('Mixed length provenance changed')
    combined=[];lengths=[];entries={}
    for source in config['training_sources']:
        original=read_jsonl(source['manifest'])
        combined.extend(original)
        lengths.extend(load_lengths(source['lengths'],source['manifest'],source_config(config,source),original))
        source_entries=json.loads(Path(source['lengths']).read_text())['entries']
        if entries.keys() & source_entries.keys():raise ValueError('Duplicate query IDs between sources')
        entries.update(source_entries)
    if combined!=rows or spec['entries']!=entries or len(entries)!=len(rows):
        raise ValueError('Mixed manifest must preserve every source row exactly once, in source order')
    return lengths


class TimestampMixtureDataset(Dataset):
    def __init__(self, config):
        self.datasets=[];self.ends=[];self.rows=[]
        self.metadata_path=Path(config['train_media_shapes'])
        for source in config['training_sources']:
            child=TimestampOptionDataset(source['manifest'],config['base'],fps=config['train_fps'],
                query_placement=config['query_placement'],metadata_path=source['media_shapes'],
                feature_root=source['feature_root'],feature_variant_root=source.get('feature_variant_root'),
                fast_feature_inputs=True)
            self.datasets.append(child);self.rows.extend(child.rows);self.ends.append(len(self.rows))
        self.lengths=load_mixture_lengths(config,self.rows)
        if self.rows!=read_jsonl(config['train']):raise ValueError('Combined manifest order mismatch')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        if isinstance(index,tuple):
            idx,*schedule=index
            return (dict(empty_pack=True) if idx is None else self[idx]) | {'pack_schedule':tuple(schedule)}
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        child=bisect_right(self.ends,index)
        offset=self.ends[child-1] if child else 0
        return self.datasets[child][index-offset]
