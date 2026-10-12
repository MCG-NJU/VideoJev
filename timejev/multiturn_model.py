"""The baseline parameterization; multiple causal readouts in each training episode."""
import torch

from .decision_model import typed_outputs
from .memory_reader import _Sample
from .timestamp_option_memory import TimestampOptionMemoryTimeJev
from .losses import decision_losses
from .decision_inputs import CHOICE, GROUNDING


class MultiturnTimeJev(TimestampOptionMemoryTimeJev):
    def __init__(self, *args, freeze_llm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_llm = bool(freeze_llm)
        if self.freeze_llm:
            self.backbone.requires_grad_(False)
            self.backbone.gradient_checkpointing_disable()
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_llm:
            self.backbone.eval()
        return self

    def _run_layers(self, *args, **kwargs):
        # Frozen-backbone warmup needs only forward activations for the head.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_llm):
            return super()._run_layers(*args, **kwargs)

    def _turn_outputs(self, logits, batch):
        args = (logits, batch['score_mask'], batch['decision_tasks'], batch.get('labels'))
        return decision_losses(*args, score_durations=batch.get('score_durations'),
            soft_iou_weight=self.soft_iou_weight)

    def forward(self, *, episode_lengths=None, turn_episode_indices=None, **batch):
        # Single-question calls use the same native causal executor.
        if episode_lengths is None:
            if batch.get('labels') is None:
                return super().forward(**batch)
            output = super().forward(**dict(batch, labels=None))
            return self._turn_outputs(output['logits'], batch)
        if self._active_video_cache is not None:
            raise ValueError('Training episodes cannot reuse persistent LLM caches')
        mask, tasks = batch['score_mask'], batch['decision_tasks']
        if ((tasks != GROUNDING) & (tasks != CHOICE)).any():
            raise ValueError('Unknown multi-turn task')
        if not episode_lengths:
            logits = self._empty_logits(mask)
        else:
            ids, types, grids = batch['input_ids'], batch['mm_token_type_ids'], batch['video_grid_thw']
            if (ids.shape[0] != 1 or batch.get('attention_mask') is not None
                    or len(episode_lengths) != len(grids)
                    or list(episode_lengths) != list(batch['packed_sample_lengths'])):
                raise ValueError('Each packed episode requires one complete native video')
            samples, offset = [], 0
            if (turn_episode_indices is None or len(turn_episode_indices) != len(mask)
                    or (turn_episode_indices < 0).any() or (turn_episode_indices >= len(grids)).any()):
                raise ValueError('Invalid turn-to-episode mapping')
            for owner, length in enumerate(episode_lengths):
                if length <= 0 or offset + length > ids.shape[1]:
                    raise ValueError('Invalid episode boundaries')
                tokens, modalities = ids[0, offset:offset+length], types[0, offset:offset+length]
                ends = ((tokens == self.vision_end_id) & (torch.roll(tokens, 1) == self.video_token_id)).nonzero().flatten()
                if len(ends) != int(grids[owner, 0]):
                    raise ValueError('Every episode must contain exactly one native video prefix')
                split = int(ends[-1]) + 1
                turns = (turn_episode_indices == owner).nonzero().flatten()
                if not len(turns):
                    raise ValueError('An episode has no supervised question')
                decisions = batch['query_indices'][turns] - offset
                candidates = batch['pair_indices'][turns] - offset
                valid = mask[turns]
                actual_options = (tokens == self.option_end_id).nonzero().flatten()
                actual_decisions = (tokens == self.decide_id).nonzero().flatten()
                if (not torch.equal(decisions, actual_decisions)
                        or not torch.equal(candidates[valid], actual_options)
                        or not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device)[None] < valid.sum(1)[:, None])
                        or not ((tasks[turns] != GROUNDING) | (valid.sum(1) == len(ends))).all()
                        or not ((tasks[turns] != CHOICE) | (valid.sum(1) >= 2)).all()
                        or not (candidates[valid] >= split).all()
                        or not ((candidates < decisions[:, None]) | ~valid).all()
                        or (len(turns) > 1 and not (candidates[1:, 0] > decisions[:-1]).all())):
                    raise ValueError('Question readouts cross turn/video boundaries')
                positions = self.backbone.compute_3d_position_ids(input_ids=tokens[None], inputs_embeds=None,
                    video_grid_thw=grids[owner:owner+1], mm_token_type_ids=modalities[None], attention_mask=None)
                samples.append(_Sample(tokens, modalities, positions, grids[owner], split,
                    candidates[0][valid[0]], int(decisions[0]), int(tasks[turns[0]]), None))
                offset += length
            pieces = self._embeddings(samples, batch['visual_features'], batch['feature_cache_ids'])
            hidden = self._run_layers(torch.cat(pieces, dim=1),
                torch.cat([x.positions for x in samples], dim=-1), episode_lengths,
                [x.split for x in samples])
            hidden = torch.cat(hidden, dim=0)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                logits = self.head(hidden[batch['pair_indices']], hidden[batch['query_indices']])
        return self._turn_outputs(logits, batch)

    def forward_with_video_memory(self, video_memory, **batch):
        if batch.get('labels') is None:
            return super().forward_with_video_memory(video_memory, **batch)
        output = super().forward_with_video_memory(video_memory, **dict(batch, labels=None))
        return self._turn_outputs(output['logits'], batch)
