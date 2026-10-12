# Evaluation

| Benchmark | Questions | Metric |
|---|---:|---|
| Charades | 3363 | mIoU, R1@0.3/0.5/0.7 |
| ActivityNet | 4500 | mIoU, R1@0.3/0.5/0.7 |
| QVHighlights | 1541 | mIoU, R1@0.3/0.5/0.7 |
| LVBench | 1549 | Accuracy |
| Video-MME | 2700 | Accuracy |
| LongVideoBench validation | 1337 | Accuracy |
| MLVU Dev | 2174 | M-Avg |

## Settings

Grounding uses 4 FPS and a default threshold of `0.25`. The 4B final plan uses 147456 tokens, smoothing 1 and `round_repair_clamp_v1`; RGB 192K uses its matching geometry. Override the threshold through `decoder_thresholds` in the config or `--decoder-threshold` when preparing a plan.

QA uses 2 FPS, at most 2048 frames, without subtitles or audio. Current scripts compare contexts 196608 and 262144 and select each benchmark's higher metric, breaking ties toward 196608. The paper uses a fixed 262144 context; see [paper settings](PAPER.md#reported-results-and-measurement-protocol).

Query groups keep manifest order and split only between complete questions, repeating the video prefix with fresh state. Oversized single questions raise an error.

## Commands

Basic 4B evaluation is in the [README](../README.md#evaluation). For QA with prepared features:

```bash
python scripts/evaluate_cached_contexts.py \
  --checkpoint /path/to/joint/hf_checkpoint \
  --plans /path/to/context_plans --feature-cache /path/to/qa_features \
  --out runs/eval4b_qa
```

For a complete 4B multi-turn plan, supply a reference bundle containing `identity.json`, `plan.json`, `decoder.json` and `inputs/`:

```bash
python scripts/prepare_multiturn_eval.py \
  --reference /path/to/input_bundle --out /path/to/plan.json
python scripts/evaluate_4b_multiturn.py \
  --checkpoint /path/to/joint/hf_checkpoint \
  --plan /path/to/plan.json --out runs/eval4b_multiturn
```

For 35B, run on each of 8 nodes with shared paths:

```bash
source scripts/env_qwen36.sh
torchrun --nnodes=8 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
  --master-port="${MASTER_PORT:-29500}" \
  scripts/evaluate_qwen36.py \
  --checkpoint /path/to/export --plans /path/to/plans --out /path/to/evaluation
```

Evaluators save predictions, metrics and completion identities. Use a new output directory when changing code, weights, inputs or protocol.
