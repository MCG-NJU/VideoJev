"""Resolution-budget variants layered over the immutable full-resolution cache."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from transformers.video_utils import VideoMetadata
from .features import cache_key, read_features, validate_cache
from .vision import prompt


def fast_cached_inputs(processor, record, query, placement='both', *, lengths_only=False):
    """Tokenize native timestamps/text once, then expand the native special token.

    Each temporal group in the skeleton has one video token. Repeating that
    special token is exactly the native processor's spatial-grid expansion; no
    visual token is pooled, removed, or re-tokenized. Parity is tested against
    cached_inputs, including multimodal type IDs.
    """
    grid=record['grid_thw']
    skeleton_grid=torch.tensor([[grid[0],2,2]],dtype=torch.long)
    metadata=VideoMetadata(total_num_frames=record['frames'],fps=record.get('effective_fps',record['fps']),frames_indices=list(range(record['frames'])))
    replacement=processor.replace_video_token(dict(video_grid_thw=skeleton_grid,video_metadata=[metadata]),video_idx=0)
    texts,_=processor.get_text_with_replacements([prompt(processor,query,placement)],[],[replacement],[])
    # Run the official token-count checks on Python integer IDs, then convert.
    # list(torch_tensor).count(id) otherwise performs one scalar Torch operation
    # per token/modality and serializes long-video preparation across threads.
    skeleton=processor(text=texts,return_tensors=None)
    skeleton.convert_to_tensors('pt')
    slots=skeleton['input_ids'][0]==processor.video_token_id
    spatial=grid[1]*grid[2]//4
    assert int(slots.sum())==grid[0] and grid[0]*spatial==record['tokens']
    length=skeleton['input_ids'].shape[1]+grid[0]*(spatial-1)
    if lengths_only:
        return length
    repeat=torch.ones(skeleton['input_ids'].shape[1],dtype=torch.long)
    repeat[slots]=spatial
    inputs={k:torch.repeat_interleave(skeleton[k],repeat,dim=1)
            for k in ['input_ids','attention_mask','mm_token_type_ids']}
    assert inputs['input_ids'].shape[1]==length
    return inputs | {'video_grid_thw':torch.tensor([grid],dtype=torch.long)}


def load_adaptive_plan(root, base_root, manifests=(), fps=2., *, require_complete=True):
    spec=validate_cache(root,[],fps,require_complete=False)
    plan=json.loads((Path(root)/'resolution_plan.json').read_text())
    base=json.loads((Path(base_root)/'provenance.json').read_text())
    if spec['base_cache_id']!=base['cache_id'] or spec['vision_sha256']!=base['vision_sha256']:
        raise ValueError('Adaptive cache base/vision identity differs')
    if require_complete:
        index=json.loads((Path(root)/'index.json').read_text())
        assert index['complete'] and index['cache_id']==spec['cache_id']
        for manifest in manifests:
            unique={}
            for s in Path(manifest).read_text().splitlines():
                row=json.loads(s);unique[(row['video_path'],row['duration'])]=row
            with ThreadPoolExecutor(max_workers=8) as pool:
                keys=list(pool.map(lambda row:cache_key(row,fps),unique.values()))
            for key in keys:
                entry=plan['entries'][key]
                assert not entry['requires_resize'] or key in index['entries']
    return spec,plan


def read_adaptive_features(base_root, variant_root, row, fps, base_id, spec, plan):
    entry=plan['entries'][cache_key(row,fps)]
    if entry['requires_resize']:
        features,record=read_features(variant_root,row,fps,spec['cache_id'])
    else:
        features,record=read_features(base_root,row,fps,base_id)
    if record['tokens']>spec['visual_token_budget'] or record['grid_thw']!=entry['grid_thw']:
        raise ValueError('Cached feature shape violates the configured resolution budget')
    return features,record
