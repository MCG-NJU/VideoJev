"""Instance-local XTuner grouped experts for the Qwen3.6 RGB experiment.

Qwen3.6-35B-A3B uses the official ``Qwen3_5Moe`` architecture.  Its
router math, shared expert and parameter names are deliberately preserved.
Routed experts use XTuner's training grouped GEMMs, including input/weight
backward kernels. Router matmul and probabilities follow XTuner's FP32 policy.
Importing this module never patches Transformers or the existing dense 4B model.
"""
from __future__ import annotations

from functools import lru_cache
from types import MethodType

import torch
from torch.nn import functional as F


def fp32_router_forward(self, hidden_states):
    """HF top-k math with XTuner FP32 routing, even inside outer BF16 AMP.

    Converting inputs with .float() alone would still let autocast downcast
    F.linear. Both logits and normalized top-k weights must retain FP32;
    expert matmuls remain BF16 and no capacity or routing objective is added.
    """
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        logits = F.linear(hidden_states.float(), self.weight.float())
        probabilities = F.softmax(logits, dtype=torch.float32, dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return logits, weights, indices


def install_fp32_routers(model):
    """Change only this model's native routed-expert gate instances."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    routers = [module for module in model.modules() if isinstance(module, Qwen3_5MoeTopKRouter)]
    if not routers:
        raise ValueError('No Qwen3_5MoeTopKRouter found in the selected model')
    for index, module in enumerate(routers):
        module.forward = MethodType(fp32_router_forward, module)
        module.timejev_router_compute_dtype = 'float32'
    return len(routers)


@lru_cache(maxsize=1)
def _xtuner_ops():
    try:
        from xtuner.v1.ops.moe.cuda import (
            cuda_token_permute,
            cuda_token_unpermute,
            triton_group_gemm,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Qwen3.6 MoE requires the XTuner CUDA grouped-GEMM backend. '
            'Source scripts/env_qwen36.sh and expose the tested XTuner checkout '
            '(set XTUNER_ROOT) with Torch 2.8/CUDA 12.8 and '
            'Triton 3.4. This experiment never falls back to an expert loop.'
        ) from exc
    if (getattr(cuda_token_permute, '__name__', None) != 'cuda_token_permute'
            or getattr(cuda_token_unpermute, '__name__', None) != 'cuda_token_unpermute'):
        raise RuntimeError(
            'XTuner selected its torch token-permute fallback because the compiled '
            'grouped_gemm CUDA extension is unavailable. Install the pinned Qwen3.6 '
            'extension in .deps-qwen36 and source scripts/env_qwen36.sh; this training '
            'path requires the fast CUDA permute/unpermute backend.'
        )
    return triton_group_gemm, cuda_token_permute, cuda_token_unpermute








def xtuner_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """HF expert interface, using two grouped GEMMs instead of E Python GEMMs.

    FP32 master parameters are allowed under CUDA BF16/FP16 autocast. Explicit
    casts are differentiable, since custom Triton GEMMs do not implement
    PyTorch's automatic autocast rules. Direct BF16 parameters need no copies.
    No expert-capacity limit, token dropping, or extra routing loss is added.
    """
    if hidden_states.device.type != 'cuda':
        raise RuntimeError('XTuner grouped Qwen3.6 experts require CUDA; CPU fallback is disabled')
    if hidden_states.ndim != 2 or top_k_index.ndim != 2:
        raise ValueError('Expected flattened [tokens, hidden] and [tokens, top_k] expert inputs')
    if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden_states.shape[0]:
        raise ValueError('Routed indices, probabilities and token counts disagree')
    if torch.is_autocast_enabled('cuda'):
        compute_dtype = torch.get_autocast_dtype('cuda')
    else:
        compute_dtype = hidden_states.dtype
    if compute_dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError('XTuner TMA grouped GEMM requires BF16/FP16 tensors or CUDA autocast')
    # ZeRO-3 gathers the direct fused Parameters at this module's pre-forward
    # hook.  Never access sliced per-expert Parameters from a different module.
    x = hidden_states.to(compute_dtype).contiguous()
    gate_up = self.gate_up_proj.to(compute_dtype).contiguous()
    down = self.down_proj.to(compute_dtype).contiguous()
    grouped_gemm, permute, unpermute = _xtuner_ops()
    indices = top_k_index.contiguous()
    counts = torch.zeros(self.num_experts, dtype=torch.int64, device=x.device)
    flat_indices = indices.reshape(-1).to(torch.int64)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
    if not x.shape[0]:
        # Keep both fused Parameters and router weights in autograd for an
        # empty diagnostic batch without launching zero-row TMA descriptors.
        return x + (gate_up.sum() + down.sum() + top_k_weights.sum()) * 0
    dispatched, row_map = permute(x, indices)
    gate, up = grouped_gemm(dispatched, gate_up, counts).chunk(2, dim=-1)
    activated = (self.act_fn(gate) * up).contiguous()
    routed_output = grouped_gemm(activated, down, counts)
    return unpermute(routed_output, row_map, top_k_weights).to(hidden_states.dtype)


def install_xtuner_grouped_experts(model, *, mark_zero3_leaf=True):
    """Patch only this model's Qwen3.5-MoE expert instances, return the count.

    This keeps HF tensor names/shapes intact for pretrained/resume/export.
    The explicit ZeRO-3 leaf flag is attached to each fused expert module;
    all expert parameters are then gathered together on every participating
    rank irrespective of that rank's selected routes.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    _xtuner_ops()  # Report missing dependencies before allocating model activations.
    modules = [module for module in model.modules() if isinstance(module, Qwen3_5MoeExperts)]
    if not modules:
        raise ValueError('No Qwen3_5MoeExperts found: this backend is only for the Qwen3.6 MoE experiment')
    for module in modules:
        module.forward = MethodType(xtuner_experts_forward, module)
        module.timejev_experts_backend = 'xtuner_triton_grouped_gemm'
        if mark_zero3_leaf:
            module._z3_leaf = True
    return len(modules)


def grouped_expert_inventory(model):
    return {
        name: {'backend': module.timejev_experts_backend,
               'zero3_leaf': bool(getattr(module, '_z3_leaf', False)),
               'num_experts': module.num_experts}
        for name, module in model.named_modules()
        if hasattr(module, 'timejev_experts_backend')
    }
