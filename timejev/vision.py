"""Official Qwen3.5 processor/model only; no custom patchify, pooling, or image path."""
import torch
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

MAX_FRAME_PIXELS = 640 * 640
VISUAL_PROTOCOL = 'qwen35_native_video_pair_midpoint_v3'


def load_processor(base):
    return AutoProcessor.from_pretrained(base)


def prompt(processor, query, query_placement='both'):
    # Reject accidental multimodal/control-token injection instead of rewriting query text.
    if '<|' in query:
        raise ValueError('Query contains model control-token syntax')
    if query_placement not in ('both', 'last'):
        raise ValueError('Unknown query placement')
    content = [
        *([dict(type='text', text='Locate the following event in the video: ' + query + '\n')] if query_placement == 'both' else []),
        dict(type='video', video='provided_in_memory'),
        dict(type='text', text='\nWhich frames match this event? ' + query),
    ]
    return processor.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=False)


def resolution_for_budget(processor, frames, height, width, visual_token_budget=None):
    """Largest per-frame cap satisfying the budget through the native resizer.

    The temporal sampling and native merge sizes are unchanged. Pixel limits are
    total-video limits in this Transformers version, including temporal rounding.
    """
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    vp = processor.video_processor
    def shape(cap):
        h, w = smart_resize(frames, height, width, temporal_factor=vp.temporal_patch_size,
                            factor=vp.patch_size * vp.merge_size,
                            min_pixels=vp.size['shortest_edge'], max_pixels=frames * cap)
        return h, w, ((frames + vp.temporal_patch_size - 1) // vp.temporal_patch_size) * (h // 32) * (w // 32)
    h, w, tokens = shape(MAX_FRAME_PIXELS)
    if visual_token_budget is None or tokens <= visual_token_budget:
        return dict(max_frame_pixels=MAX_FRAME_PIXELS, processed_hw=[h,w], visual_tokens=tokens)
    if visual_token_budget <= 0:
        raise ValueError('Visual token budget must be positive')
    # Never truncate frames/tokens. An impossible native minimum is an error.
    lo = max(1024, (vp.size['shortest_edge'] + frames - 1) // frames)
    hi = MAX_FRAME_PIXELS
    if shape(lo)[2] > visual_token_budget:
        raise ValueError('Visual budget cannot retain all frames at the native minimum resolution')
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if shape(mid)[2] <= visual_token_budget:
            lo = mid
        else:
            hi = mid - 1
    h, w, tokens = shape(lo)
    return dict(max_frame_pixels=lo, processed_hw=[h,w], visual_tokens=tokens)


def prepare_video(processor, sample, query, query_placement='both', *, visual_token_budget=None):
    video = sample['video']
    n, height, width, _ = video.shape
    resolution = resolution_for_budget(processor, n, height, width, visual_token_budget)
    # This installed official video processor takes a TOTAL video pixel budget.
    # N * per-frame cap prevents it from reducing resolution as video length grows.
    metadata = VideoMetadata(total_num_frames=n, fps=sample['fps'], frames_indices=list(range(n)))
    inputs = processor(
        text=[prompt(processor, query, query_placement)], videos=[video], video_metadata=[metadata],
        do_sample_frames=False,
        size=dict(shortest_edge=processor.video_processor.size['shortest_edge'],
                  longest_edge=n * resolution['max_frame_pixels']),
        return_tensors='pt')
    grid = inputs['video_grid_thw'][0]
    resized_h, resized_w = int(grid[1]) * 16, int(grid[2]) * 16
    assert resized_h * resized_w <= MAX_FRAME_PIXELS, (resized_h, resized_w)
    assert int(grid[0]) == (n + 1) // 2
    expected_tokens = int(grid.prod()) // 4
    assert expected_tokens == resolution['visual_tokens']
    if visual_token_budget is not None:
        assert expected_tokens <= visual_token_budget
    assert int((inputs['input_ids'] == processor.video_token_id).sum()) == expected_tokens
    # Only model-forward tensors. VideoMetadata remains an audit record, not a model input.
    allowed = ['input_ids', 'attention_mask', 'mm_token_type_ids',
               'pixel_values_videos', 'video_grid_thw']
    tensors = {k: inputs[k] for k in allowed}
    audit = dict(query_placement=query_placement, original_hw=[height, width], processed_hw=[resized_h, resized_w],
                 sample_fps=sample['fps'], input_source=sample.get('input_source', 'video'),
                 sampled_frames=n, temporal_groups=int(grid[0]), visual_tokens=expected_tokens,
                 pixels_per_frame=resized_h * resized_w, max_frame_pixels=resolution['max_frame_pixels'],
                 visual_protocol=VISUAL_PROTOCOL, media_duration=sample['media_duration'],
                 sampled_duration=sample['sampled_duration'],
                 ignored_duplicate_pts=sample['ignored_duplicate_pts'],
                 decoded_times=sample['decoded_times'].tolist())
    if visual_token_budget is not None:
        audit['visual_token_budget'] = visual_token_budget
    return tensors, audit
