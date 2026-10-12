"""Complete video/query sample metadata and inference-cache preconditions."""
from dataclasses import dataclass
import torch
from .decision_model import DecisionTimeJev


@dataclass
class _Sample:
    ids: torch.Tensor
    types: torch.Tensor
    positions: torch.Tensor
    grid: torch.Tensor
    split: int
    pairs: torch.Tensor
    query: int
    task: int
    video_identity: object


class MemoryReaderTimeJev(DecisionTimeJev):
    """Shared base for the native causal video executor."""
    def _require_cache_mode(self):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Video caching requires model.eval() and torch.no_grad()')
