# Custom data

Use JSONL, one question per line:

```jsonl
{"id":"v1/0","video_key":"v1","video_path":"videos/v1.mp4","duration":10.0,"task":"grounding","query":"A person opens the door.","spans":[[2.0,4.5]]}
{"id":"v2/0","video_key":"v2","video_path":"videos/v2.mp4","duration":12.0,"task":"choice","query":"What does the person pick up?","options":["A. A cup.","B. A book."],"answer_index":0}
```

Paths are relative to the JSONL file. Times are in seconds; `answer_index` is zero-based. Keep IDs unique and use one `video_key` for all questions about the same video. Either task or a mixture is supported. Two-stage training needs at least two videos.

## Prepare

```bash
source scripts/env.sh
python scripts/prepare_data.py \
  --manifest /path/to/train.jsonl --base /path/to/Qwen3.5-4B \
  --out data/custom4b --world-size 8
```

The command splits videos 30%/70% between head warmup and joint training, preserves question order, and generates RGB metadata, token lengths, packs, `ready.json`, and training configs. Use `--head-fraction` to change the split; `--world-size` is the total number of training GPUs and must divide 64.

For 35B, use `--model-size 35B-A3B --base /path/to/Qwen3.6-35B-A3B` and a separate output directory. Preparation uses the tokenizer and video files; no feature cache is required.

## Train

For the 8-GPU configuration above:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config data/custom4b/configs/head_only.json --out runs/custom4b/head_only

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config data/custom4b/configs/joint.json --out runs/custom4b/joint \
  --init-head runs/custom4b/head_only/head.pt
```

For 35B, source `scripts/env_qwen36.sh` and use `scripts/train_qwen36.py` with its generated configs. For multiple nodes, use the corresponding `torchrun` node and rendezvous arguments.

## Evaluate

Use the same labeled JSONL format with the [HF evaluator](EVALUATION.md). Official QA benchmarks additionally retain `uid` for LVBench or `question_id` for the other three benchmarks; MLVU also needs `task_type` and `official_duration`. Preserve the official question and option order.
