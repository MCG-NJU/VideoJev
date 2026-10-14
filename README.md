<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme-header-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/readme-header.svg">
    <img src="docs/assets/readme-header.svg" alt="VideoJev" width="500">
  </picture>
</p>

<h3 align="center">System One Model for Video Understanding and Interaction</h3>

<p align="center">
  <a href="https://mcg-nju.github.io/VideoJev"><img src="https://img.shields.io/badge/VideoJev-Homepage-555555?labelColor=c96343" alt="VideoJev Homepage" height="28"></a>
  <a href="https://arxiv.org/abs/2610.18615"><img src="https://img.shields.io/badge/arXiv-2610.18615-555555?labelColor=c96343" alt="arXiv: 2610.18615" height="28"></a>
  <a href="https://huggingface.co/ZhuYuhan/VideoJev-4B"><img src="https://img.shields.io/badge/Hugging%20Face-VideoJev--4B-555555?labelColor=b88a45" alt="Hugging Face · VideoJev-4B" height="28"></a>
  <a href="https://huggingface.co/ZhuYuhan/VideoJev-35B-A3B"><img src="https://img.shields.io/badge/Hugging%20Face-VideoJev--35B--A3B-555555?labelColor=778269" alt="Hugging Face · VideoJev-35B-A3B" height="28"></a>
</p>

<p align="center">
  <a href="#overview">🎬 Overview</a> ·
  <a href="#results">📊 Results</a> ·
  <a href="#models">🤗 Models</a> ·
  <a href="#quick-start">🚀 Quick start</a> ·
  <a href="#inference-output">💬 Outputs</a> ·
  <a href="#training">🏋️ Training</a> ·
  <a href="#evaluation">🧪 Evaluation</a>
</p>

<p align="center">
  <img src="docs/assets/videojev-teaser.png" alt="VideoJev paper teaser: shared candidate scoring and video computation, with grounding and QA quality versus model forward latency" width="100%">
</p>

---

## Overview

**VideoJev** performs temporal grounding and multiple-choice video QA through direct candidate scoring, without text generation.

[![Watch the VideoJev overview](https://mcg-nju.github.io/VideoJev/assets/videojev-promo-poster.jpg)](https://mcg-nju.github.io/VideoJev/#video)

<details>
<summary><strong>Architecture</strong></summary>

![VideoJev architecture](docs/assets/videojev-method.png)

</details>

## Results

### Temporal grounding

![Paper Table 1: temporal grounding](docs/assets/temporal-grounding-table.png)

### Video question answering

![Paper Table 2: video question answering](docs/assets/video-qa-table.png)

## Models

| Model | Base model | Weight format | Usage |
|---|---|---|---|
| **VideoJev-4B** | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | BF16 · Safetensors | [Checkpoint & examples ↗](https://huggingface.co/ZhuYuhan/VideoJev-4B) |
| **VideoJev-35B-A3B** | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | BF16 · Safetensors | [Checkpoint & examples ↗](https://huggingface.co/ZhuYuhan/VideoJev-35B-A3B) |

## Quick start

Set up the [environment](#environment), then run the 10-second demo:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

video = hf_hub_download(
    repo_id="ZhuYuhan/VideoJev-Demo", filename="eating_spaghetti.mp4",
    repo_type="dataset", revision="7e2b9685bca592c3b929268f6fcf82c4ad74b267",
)
repo = "ZhuYuhan/VideoJev-4B"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, attn_implementation="flash_attention_2",
)
model.prepare_for_inference()

questions = [
    {"id": "grounding-1", "task": "grounding",
     "query": "The woman in green eats pasta."},
    {"id": "qa-1", "task": "choice",
     "query": "What is the woman in green eating?",
     "options": ["A. Pasta.", "B. Rice.", "C. Soup."]},
]
results = model.predict_video(
    processor, video, questions,
    fps=2.0, max_frames=2048, context=8192,
)
print(json.dumps(results, indent=2))
```

Use `ZhuYuhan/VideoJev-35B-A3B` for the larger model.

## Inference output

Recorded 4B output:

```json
[
  {
    "id": "grounding-1",
    "task": "grounding",
    "probabilities": [
      0.005761092994362116,
      0.010865824297070503,
      0.026081746444106102,
      0.9046059250831604,
      0.9879851937294006,
      0.9887627959251404,
      0.9896195530891418,
      0.9849089980125427,
      0.833530068397522,
      0.33037981390953064
    ],
    "interval": [
      3.0,
      10.0
    ]
  },
  {
    "id": "qa-1",
    "task": "choice",
    "probabilities": [
      0.9999161958694458,
      5.755242091254331e-05,
      2.617507561808452e-05
    ],
    "answer_index": 0,
    "answer": "A. Pasta."
  }
]
```

`interval` is in seconds; `answer_index` is zero-based. Probabilities use sigmoid for grounding and softmax for QA.

Grounding defaults to `threshold=0.25`. Pass `threshold=...` to `predict_video` or `predict_features` to override it.

## Environment

Python 3.12, PyTorch 2.8 / CUDA 12.8, Triton 3.4, FlashAttention 2 and causal-conv1d. Install the matching CUDA packages first:

```bash
pip install -r requirements.txt
source scripts/env.sh
```

Training requires [XTuner](https://github.com/InternLM/xtuner/tree/76e705134521eff867b409f3b3451df1c4d8dd36); 35B also requires its compiled grouped-GEMM/token-permutation extension. Set `XTUNER_ROOT` before sourcing the environment, or install XTuner in it.

Put the base models in `models/Qwen3.5-4B` and `models/Qwen3.6-35B-A3B`, or set `VIDEOJEV_BASE_4B` and `VIDEOJEV_BASE_35B`.

## Data

[Prepare your JSONL data](docs/DATA.md); the command generates ready-to-train configs for either model.

## Training

| Stage | Trainable parameters | Grounding loss | QA loss |
|---|---|---|---|
| Head warmup | Head | BCE | CE |
| Joint | Language backbone + head | BCE + 2 × duration-weighted SoftIoU | CE |

Vision stays frozen. Each stage runs for one epoch with a global batch of 64 packs.

### Launch

Use the configs generated by [data preparation](docs/DATA.md):

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config data/custom4b/configs/head_only.json --out runs/custom4b/head_only

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config data/custom4b/configs/joint.json --out runs/custom4b/joint \
  --init-head runs/custom4b/head_only/head.pt
```

For 35B-A3B, use `scripts/train_qwen36.py` with its generated configs and `scripts/env_qwen36.sh`. See [Data & training](docs/DATA.md) for preparation and GPU settings.

### Checkpoints

4B exports **`joint/hf_checkpoint/`** in Hugging Face format, preserving the base model's dtype (BF16). See [Checkpoint export](docs/CHECKPOINTS.md) for recovery files and standalone 35B conversion.

## Evaluation

```bash
source scripts/env.sh
python scripts/evaluate_hf.py \
  --checkpoint ZhuYuhan/VideoJev-4B \
  --manifest /path/to/eval.jsonl --out runs/eval4b
```

The same command supports `ZhuYuhan/VideoJev-35B-A3B`. QA uses a fixed 256K context. See [Evaluation](docs/EVALUATION.md) for benchmarks and distributed runs.

## Verification

```bash
source scripts/env.sh
python -m pytest -q tests
```

Set `VIDEOJEV_TEST_MODEL=/path/to/Qwen3.5-4B` for tokenizer checks. Tests skip when required local inputs are absent.

## Documentation

[Paper](https://arxiv.org/abs/2610.18615) · [Data](docs/DATA.md) · [Evaluation](docs/EVALUATION.md) · [Checkpoint export](docs/CHECKPOINTS.md)

## Citation

```bibtex
@misc{zhu2026videojevmodelvideounderstanding,
  title={VideoJev: System One Model for Video Understanding and Interaction},
  author={Yuhan Zhu and Xiangyu Zeng and Qingyi Si and Chenxu Yang and Yifei Li and Luozheng Qin and Junhao Zhou and Dingyu Yao and Jun Zhang and Chuanyu Qin and Zhiqiu Zhang and Yuandong Yang and Yuqi Xu and Tianyu Zhang and Hongcheng Guo and Jiaqi Wang and Nan Duan and Limin Wang},
  year={2026},
  eprint={2610.18615},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.18615},
}
```

---

### 🤝 Acknowledgements

Thanks to the teams behind [**Qwen**](https://huggingface.co/Qwen) for the open model checkpoints, [**XTuner**](https://github.com/InternLM/xtuner) for the training framework, and [**TimeLens-100K**](https://huggingface.co/datasets/TencentARC/TimeLens-100K) and [**TimeLens2-93K**](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K) for the open training data.

<p align="center">
  <a href="LICENSE">Apache 2.0</a> · <a href="NOTICE">NOTICE</a>
</p>
