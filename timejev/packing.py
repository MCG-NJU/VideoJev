"""XTuner whole-dataset packing, with native Qwen grounding boundaries."""
import math
import random
import json
from pathlib import Path
import torch
from torch.utils.data import Sampler


def load_lengths(path, manifest, config, rows):
    from .engine import sha256
    from .feature_sources import feature_sources
    spec=json.loads(Path(path).read_text())
    root,variant_root=feature_sources(config,training=Path(manifest).resolve()==Path(config['train']).resolve())
    variant=json.loads((Path(variant_root or root)/'provenance.json').read_text())
    if (spec['manifest_sha256']!=sha256(manifest) or spec['cache_id']!=variant['cache_id']
            or variant['max_sequence_length']!=config['max_sequence_length']
            or spec['query_placement']!=config.get('query_placement','both')
            or spec['max_sequence_length']!=config['max_sequence_length']):
        raise ValueError('Packed token-length manifest belongs to another dataset/config/cache')
    assert len(spec['entries'])==len(rows)
    return [spec['entries'][r['id']]['tokens'] for r in rows]


def best_fit(indices, lengths, capacity):
    packs, sizes = [], []
    for idx in sorted(indices, key=lambda i: lengths[i], reverse=True):
        n = lengths[idx]
        if n > capacity or n <= 0:
            raise ValueError(f'Sample {idx} has {n} tokens, outside (0, {capacity}]')
        available = [j for j,s in enumerate(sizes) if s+n <= capacity]
        if available:
            j = max(available, key=lambda j:sizes[j])
            packs[j].append(idx); sizes[j] += n
        else:
            packs.append([idx]); sizes.append(n)
    return packs


def build_xtuner_packs(lengths, capacity, seed):
    """Call XTuner's actual soft-pack dataset on the full original dataset."""
    import numpy as np
    from xtuner.v1.datasets.packing import ExpandSoftPackDataset

    if not lengths or min(lengths) <= 0 or max(lengths) > capacity:
        raise ValueError('Every complete query must fit the pack capacity')

    class LengthDataset:
        num_tokens = np.asarray(lengths, dtype=np.int64)
        proxy_attn_flops = num_tokens * (1.0 + num_tokens / 32768.0)
        def __len__(self): return len(lengths)
        def __getitem__(self, i): return i

    dataset = ExpandSoftPackDataset([LengthDataset()], pack_max_length=capacity,
        pack_workers=1, pack_chunk_size=10000, pack_extra_buffer_size=1000, seed=seed)
    packs = [list(map(int, ids)) for ids in dataset.pack_infos['indices']]
    flat = [i for pack in packs for i in pack]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in pack) <= capacity for pack in packs)
    return packs


def xtuner_source_hashes():
    from xtuner.v1.datasets import packing, sampler
    from .engine import sha256
    return {str(Path(m.__file__).resolve()): sha256(m.__file__) for m in [packing, sampler]}


class PackedBatchSampler(Sampler):
    """Global batch counts packed sequences, using XTuner packing and sampling.

    Packs are built over the entire dataset before forming optimizer batches.
    XTuner LengthGroupedSampler balances those packs by its attention-cost proxy. Its terminal repeated slots
    are replaced by zero-loss placeholders so every original query appears once;
    the last partial update retains every remaining complete pack.
    """
    def __init__(self, lengths, capacity, rank=0, world=1, seed=0, global_batch=32, *, packs=None):
        if global_batch < 1 or world < 1 or global_batch % world or not 0 <= rank < world:
            raise ValueError('Packed global batch must be divisible by data parallel size')
        if not lengths or min(lengths) <= 0 or max(lengths) > capacity:
            raise ValueError('Empty dataset or sample outside pack capacity')
        self.lengths, self.capacity = lengths, capacity
        self.rank, self.world, self.seed = rank, world, seed
        self.global_batch, self.epoch = global_batch, 0
        self.packs = build_xtuner_packs(lengths, capacity, seed) if packs is None else packs
        assert sorted(i for pack in self.packs for i in pack) == list(range(len(lengths)))
        assert all(pack and sum(lengths[i] for i in pack) <= capacity for pack in self.packs)
        self.total_updates = math.ceil(len(self.packs) / global_batch)
        self._cached = None

    def plan(self):
        from xtuner.v1.datasets.sampler import LengthGroupedSampler
        if self._cached is not None and self._cached[0] == self.epoch:
            return self._cached[1]

        class RankMesh:
            def __init__(mesh, rank): mesh.rank = rank
            def size(mesh): return self.world
            def get_local_rank(mesh): return mesh.rank

        class PackDataset:
            longest = [sum(self.lengths[i]*(1+self.lengths[i]/32768) for i in p) for p in self.packs]
            def __len__(dataset): return len(self.packs)

        rank_orders = []
        for rank in range(self.world):
            sampler = LengthGroupedSampler(PackDataset(), global_batch_size=self.global_batch,
                dp_mesh=RankMesh(rank), seed=self.seed, round_up=True)
            sampler.set_epoch(self.epoch)
            rank_orders.append(list(sampler))
        order = [rank_orders[r][i] for i in range(len(rank_orders[0])) for r in range(self.world)]
        # Everything beyond this point is XTuner's round-up padding, not new data.
        order = order[:len(self.packs)]
        assert sorted(order) == list(range(len(self.packs)))
        plan = []
        for update, start in enumerate(range(0, len(order), self.global_batch)):
            batch = order[start:start+self.global_batch]
            global_ids = [i for pack_id in batch for i in self.packs[pack_id]]
            rounds = math.ceil(len(batch)/self.world)
            for micro in range(rounds):
                slots = batch[micro*self.world:(micro+1)*self.world]
                slots += [None]*(self.world-len(slots))
                plan.append(dict(update=update+1, micro=micro, rounds=rounds,
                    global_ids=global_ids, global_pack_ids=batch, rank_pack_ids=slots,
                    rank_ids=[self.packs[i] if i is not None else [] for i in slots]))
        self._cached = (self.epoch, plan)
        return plan

    def __iter__(self):
        for entry in self.plan():
            ids = entry['rank_ids'][self.rank]
            meta = (entry['update'], entry['micro'], entry['rounds'],
                    len(entry['global_ids']), len(entry['global_pack_ids']))
            yield [(i,*meta) for i in ids] if ids else [(None,*meta)]

    def __len__(self):
        return len(self.plan())


class PackedCollator:
    def __init__(self, tokenizer, max_length, *, pad_to_max_length=False):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.pad_to_max_length=pad_to_max_length
        self.pad = tokenizer.pad_token_id
        self.vend=tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.video=tokenizer.convert_tokens_to_ids('<|video_pad|>')
        self.im_end=tokenizer.convert_tokens_to_ids('<|im_end|>')

    def __call__(self, items):
        schedule=items[0].get('pack_schedule')
        assert all(x.get('pack_schedule')==schedule for x in items)
        items=[x for x in items if not x.get('empty_pack',False)]
        ns=len(items)
        lengths=[x['inputs']['input_ids'].shape[1] for x in items]
        real_tokens=sum(lengths)
        total=max(64,(real_tokens+63)//64*64)
        if self.pad_to_max_length and ns:
            total=max(total,self.max_length)
        if total>self.max_length:
            raise ValueError(f'Pack has {total} tokens, capacity {self.max_length}; no truncation allowed')
        ids=torch.full((1,total),self.pad,dtype=torch.long)
        types=torch.zeros_like(ids)
        groups=max([len(x['times']) for x in items],default=1)
        pairs=torch.zeros((max(ns,1),groups),dtype=torch.long)
        queries=torch.zeros(max(ns,1),dtype=torch.long)
        labels=torch.zeros((max(ns,1),groups))
        mask=torch.zeros((max(ns,1),groups),dtype=torch.bool)
        offset=0
        for i,(x,n) in enumerate(zip(items,lengths)):
            src=x['inputs']['input_ids'][0]
            assert x['inputs']['attention_mask'].all()
            ids[0,offset:offset+n]=src
            types[0,offset:offset+n]=x['inputs']['mm_token_type_ids'][0]
            end=((src==self.vend)&(torch.roll(src,1)==self.video)).nonzero().flatten()
            assert len(end)==len(x['times'])
            pairs[i,:len(end)]=end+offset
            queries[i]=(src==self.im_end).nonzero().flatten()[-1]+offset
            labels[i,:len(end)]=x['labels']; mask[i,:len(end)]=True
            offset+=n
        # XTuner uses independent short padding chunks to avoid quadratic attention
        # or recurrent state crossing into padding. Grounding keeps every input token.
        from xtuner.v1.datasets.collator import split_to_chunks
        all_lengths=lengths+split_to_chunks(total-real_tokens,256)
        cu=torch.tensor([0,*torch.tensor(all_lengths).cumsum(0).tolist()],dtype=torch.int32)
        seq_idx=torch.repeat_interleave(torch.arange(len(all_lengths),dtype=torch.int32),torch.tensor(all_lengths))[None,:]
        result=dict(input_ids=ids,attention_mask=None,mm_token_type_ids=types,
                    video_grid_thw=torch.cat([x['inputs']['video_grid_thw'] for x in items]) if ns else torch.empty((0,3),dtype=torch.long),
                    visual_features=torch.cat([x['inputs']['visual_features'] for x in items]) if ns else torch.empty((0,2560),dtype=torch.bfloat16),
                    feature_cache_ids=[x['audit']['feature_cache_id'] for x in items],
                    pair_indices=pairs,query_indices=queries,labels=labels,score_mask=mask,
                    packed_sample_lengths=lengths,cu_seq_lens_q=cu,cu_seq_lens_k=cu,
                    max_length_q=max(all_lengths),max_length_k=max(all_lengths),seq_idx=seq_idx,
                    sample_count=ns,real_tokens=real_tokens,pack_schedule=schedule,
                    rows=[x['row'] for x in items],times=[x['times'] for x in items],audits=[x['audit'] for x in items])
        return result
