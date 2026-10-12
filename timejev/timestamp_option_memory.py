"""Grounding timestamp text candidates, with native asymmetric video caching."""
import torch

from .asymmetric_video_memory import AsymmetricVideoMemoryTimeJev, _fork_prefix_cache
from .decision_inputs import OPTION_END, DECIDE, GROUNDING, CHOICE
from .memory_reader import _Sample
from .timestamp_option_inputs import GROUNDING_CANDIDATES, INPUT_PROTOCOL


class TimestampOptionMemoryTimeJev(AsymmetricVideoMemoryTimeJev):
    experiment_arch = 'timestamp_option_memory'

    def __init__(self, base, *args, grounding_candidates=GROUNDING_CANDIDATES, **kwargs):
        if grounding_candidates != GROUNDING_CANDIDATES:
            raise ValueError('Timestamp-option memory requires suffix timestamp candidates')
        super().__init__(base, *args, **kwargs)
        self.grounding_candidates = grounding_candidates
        self.option_end_id = self.decision_metadata['tokens'][OPTION_END]
        self.decide_id = self.decision_metadata['tokens'][DECIDE]
        self.decision_metadata['grounding_candidates'] = grounding_candidates
        self.decision_metadata['input_protocol'] = INPUT_PROTOCOL
        self.decision_metadata['architecture'].update(
            grounding_candidate_readout='suffix OPTION_END for each native timestamp',
            grounding_probability='independent sigmoid; multi-label BCE',
            choice_candidate_readout='suffix OPTION_END; categorical softmax/CE',
            timestamp_display='official video-prefix timestamp text copied without changing label coordinates')

    def _samples(self, input_ids, attention_mask, mm_token_type_ids, video_grid_thw,
                 pair_indices, query_indices, score_mask, decision_tasks,
                 packed_sample_lengths=None, rows=None, audits=None, **unused):
        if packed_sample_lengths is None:
            if attention_mask is None:
                raise ValueError('Unpacked timestamp inputs require explicit padding masks')
            lengths = attention_mask.sum(1).tolist()
            if len(video_grid_thw) == 0:
                return []
        else:
            if input_ids.shape[0] != 1 or attention_mask is not None:
                raise ValueError('Packed timestamp inputs require one padding-free token stream')
            lengths = list(packed_sample_lengths)
        if len(lengths) != len(video_grid_thw):
            raise ValueError('Each timestamp-option sample requires one complete native video')
        grids, tasks, queries = video_grid_thw.cpu().tolist(), decision_tasks.tolist(), query_indices.tolist()
        samples, offset = [], 0
        for index, length in enumerate(lengths):
            row, start = (0, offset) if packed_sample_lengths is not None else (index, 0)
            ids = input_ids[row, start:start+length]
            types = mm_token_type_ids[row, start:start+length]
            ends = ((ids == self.vision_end_id) & (torch.roll(ids, 1) == self.video_token_id)).nonzero().flatten()
            if len(ends) != grids[index][0]:
                raise ValueError('Native video temporal groups differ from grid metadata')
            split = int(ends[-1])+1
            pairs = pair_indices[index][score_mask[index]]-start
            query = queries[index]-start
            options = (ids == self.option_end_id).nonzero().flatten()
            if (not split <= query < length or ids[query] != self.decide_id or not len(pairs)
                    or not torch.equal(pairs, options) or not ((pairs >= split) & (pairs < query)).all()):
                raise ValueError('All candidates must be complete suffix options preceding DECIDE')
            if tasks[index] == GROUNDING:
                if len(pairs) != grids[index][0]:
                    raise ValueError('Every native temporal group requires one timestamp option')
            elif tasks[index] != CHOICE or len(pairs) < 2:
                raise ValueError('Unknown task or invalid choice options')
            positions = self.backbone.compute_3d_position_ids(input_ids=ids[None], inputs_embeds=None,
                video_grid_thw=video_grid_thw[index:index+1], mm_token_type_ids=types[None], attention_mask=None)
            identity = audits[index].get('feature_path') if audits else None
            if identity is None and rows:
                identity = next((rows[index][key] for key in ('video', 'video_path', 'path') if key in rows[index]), None)
            samples.append(_Sample(ids, types, positions, video_grid_thw[index], split,
                pairs, query, tasks[index], identity))
            offset += length
        return samples

    def _empty_logits(self, score_mask):
        embedding = self.backbone.get_input_embeddings()
        ids = torch.full((1, 64), self.vision_end_id, device=embedding.weight.device, dtype=torch.long)
        positions = torch.arange(64, device=ids.device)[None, None].expand(3, 1, -1)
        hidden = self._run_layers(embedding(ids), positions, [64], [32])[0]
        with torch.autocast(device_type=ids.device.type, enabled=False):
            zero = self.head(hidden[62][None, None], hidden[63][None]).sum()*0.
        return zero.expand(score_mask.shape)

    def _cached_logits(self, samples, memories, score_mask):
        candidates, queries = [], []
        for item, memory in zip(samples, memories):
            embeds = self.backbone.get_input_embeddings()(item.ids[None, item.split:])
            hidden = self._run_layers(embeds, item.positions[:, :, item.split:],
                [len(item.ids)-item.split], [0], native_cache=_fork_prefix_cache(memory.native_cache))[0]
            # Both tasks read text options. The task index remains unchanged:
            # grounding retains independent sigmoid/BCE, QA softmax/CE.
            candidates.append(hidden[item.pairs-item.split])
            queries.append(hidden[item.query-item.split])
        return self._score_hidden(samples, candidates, queries, score_mask)

    def load_checkpoint(self, checkpoint_data):
        if checkpoint_data.get('config', {}).get('grounding_candidates') != self.grounding_candidates:
            raise ValueError('Checkpoint uses another grounding candidate protocol')
        super().load_checkpoint(checkpoint_data)
