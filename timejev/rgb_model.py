"""Online frozen vision and a rank-invariant module schedule for ZeRO-3."""
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from .multiturn_model import MultiturnTimeJev
from .rgb_data import PROTOCOL


def device_chunks(chunks, device):
    """Two-chunk H2D prefetch; CPU preprocessing is separately worker-prefetched."""
    if device.type != 'cuda':
        yield from (x.to(device) for x in chunks)
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(chunks)
    def copy_next():
        cpu = next(iterator, None)
        if cpu is None:
            return None
        with torch.cuda.stream(stream):
            return cpu.pin_memory().to(device, non_blocking=True)
    pending = copy_next()
    while pending is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        current = pending
        current.record_stream(torch.cuda.current_stream(device))
        pending = copy_next()
        yield current


class ChunkedRGBVision(Qwen3_5VisionModel):
    """One ZeRO leaf call per pack, with bounded native ViT chunks inside it."""
    @torch.no_grad()
    def forward(self, video_pixel_chunks, grid_thw):
        device = next(self.parameters()).device
        grids = grid_thw.cpu().tolist()
        if len(grids) != len(video_pixel_chunks):
            raise ValueError('Each RGB episode needs its complete patch chunks')
        output = []
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for chunks, (t, h, w) in zip(video_pixel_chunks, grids):
                groups = 0
                for pixels in device_chunks(chunks, device):
                    if len(pixels) % (h*w):
                        raise ValueError('RGB patch chunks cut native temporal groups')
                    count = len(pixels)//(h*w)
                    grid = torch.tensor([[count, h, w]], device=device, dtype=torch.long)
                    encoded = super().forward(pixels, grid_thw=grid, return_dict=True).pooler_output
                    if len(encoded) != count*h*w//4:
                        raise ValueError('Online native vision token count differs from metadata')
                    output.append(encoded)
                    groups += count
                if groups != t:
                    raise ValueError('Missing/duplicated RGB temporal groups')
        return (torch.cat(output) if output else
                torch.empty((0, self.config.out_hidden_size), device=device, dtype=torch.bfloat16))


class RGBMultiturnTimeJev(MultiturnTimeJev):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone.visual.__class__ = ChunkedRGBVision
        self.feature_cache_ids.add(PROTOCOL)
        self.decision_metadata['rgb_protocol'] = PROTOCOL

    def forward(self, *, video_pixel_chunks=None, **batch):
        if video_pixel_chunks is not None or ('episode_lengths' in batch and 'visual_features' not in batch):
            # Empty trailing pack slots must enter the same visual leaf, lookup,
            # decoder layers and head as all other ranks. No per-video collectives.
            chunks = video_pixel_chunks if video_pixel_chunks is not None else []
            batch['visual_features'] = self.backbone.visual(chunks, grid_thw=batch['video_grid_thw'])
            batch['feature_cache_ids'] = [PROTOCOL]*len(batch['video_grid_thw'])
        return super().forward(**batch)

    def _embeddings(self, samples, visual_features, feature_cache_ids, *, prefix_only=False):
        if (visual_features is None or feature_cache_ids != [PROTOCOL]*len(samples)
                or not self._packed_kernels_available):
            raise ValueError('RGB execution requires current online native features')
        ids = [s.ids[:s.split] if prefix_only else s.ids for s in samples]
        # Calling Embedding once per video would make ZeRO-3 gather counts differ
        # across ranks with different numbers of packed videos.
        tokens = torch.cat(ids)[None]
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_llm):
            embeds = self.backbone.get_input_embeddings()(tokens)
            mask = tokens == self.video_token_id
            if int(mask.sum()) != len(visual_features):
                raise ValueError('Every native video token requires one online feature')
            embeds = embeds.masked_scatter(mask[..., None].expand_as(embeds), visual_features.to(embeds))
        return list(embeds.split([len(x) for x in ids], dim=1))

    def _empty_logits(self, score_mask):
        # Do not inspect embedding.weight outside Embedding.forward: ZeRO would
        # register a new external parameter only on ranks with empty tail slots.
        ids = torch.full((1, 64), self.vision_end_id, device=score_mask.device, dtype=torch.long)
        positions = torch.arange(64, device=ids.device)[None, None].expand(3, 1, -1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_llm):
            embeds = self.backbone.get_input_embeddings()(ids)
        hidden = self._run_layers(embeds, positions, [64], [32])[0]
        with torch.autocast(device_type=ids.device.type, enabled=False):
            zero = self.head(hidden[62][None, None], hidden[63][None]).sum()*0.
        return zero.expand(score_mask.shape)


def configure_zero3(model):
    from deepspeed.utils import set_z3_leaf_modules
    # Chunk counts vary by video; child gathers must stay inside one leaf call.
    set_z3_leaf_modules(model, [ChunkedRGBVision])


def consolidated_state(module):
    """Collect FP32 export on rank zero, gathering one parameter at a time.

    Called by EVERY rank. A plain state_dict() contains empty ZeRO-3 shards.
    """
    import contextlib
    import deepspeed
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    result = {} if rank == 0 else None
    for name, parameter in module.named_parameters(remove_duplicate=False):
        context = (deepspeed.zero.GatheredParameters([parameter], modifier_rank=None)
                   if hasattr(parameter, 'ds_id') else contextlib.nullcontext())
        with context:
            if rank == 0:
                result[name] = parameter.detach().cpu().clone()
    if rank == 0:
        # Include only persistent buffers, respecting state_dict's semantics.
        parameters = dict(module.named_parameters(remove_duplicate=False))
        for name, value in module.state_dict().items():
            if name not in parameters:
                result[name] = value.detach().cpu().clone()
    return result
