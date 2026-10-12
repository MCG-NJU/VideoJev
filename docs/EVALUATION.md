# Evaluation

Both released HF checkpoints work with the same evaluator:

```bash
source scripts/env.sh
python scripts/evaluate_hf.py \
  --checkpoint MCG-NJU/VideoJev-4B \
  --manifest /path/to/eval.jsonl --out runs/eval4b
```

Use `MCG-NJU/VideoJev-35B-A3B` or a local HF directory for `--checkpoint`. See the [JSONL format](DATA.md). Results are saved as `predictions.jsonl`, `metrics.json`, and `summary.json`.

Grounding uses **4 FPS**, **147456 tokens**, and **threshold 0.25**; change the threshold with `--threshold`. QA uses **2 FPS**, **262144 tokens**, at most **2048 frames**, and no subtitles or audio. Questions share a video prefix and split only between complete questions.

## Official benchmarks

Place each benchmark under `data/benchmarks/<name>/` with `questions.jsonl`, video files, and its official QA annotations (`official.json`; LVBench uses `official.jsonl`).

```bash
python scripts/evaluate_hf.py \
  --checkpoint MCG-NJU/VideoJev-4B \
  --benchmark all --data-root data/benchmarks --out runs/benchmarks
```

| Name | Questions | Metric |
|---|---:|---|
| `charades` | 3363 | mIoU, R1@0.3/0.5/0.7 |
| `activitynet` | 4500 | mIoU, R1@0.3/0.5/0.7 |
| `qvhighlights` | 1541 | mIoU, R1@0.3/0.5/0.7 |
| `lvbench` | 1549 | Accuracy |
| `videomme` | 2700 | Accuracy |
| `longvideobench` | 1337 | Accuracy |
| `mlvu` | 2174 | M-Avg |

Use `--benchmark <name>` for one benchmark, `--reference` for a different official annotation path, or `--plans` for existing `evaluation_<context>.json` RGB plans. `--dry-run` checks inputs without loading weights. With `torchrun`, each GPU evaluates complete videos independently.

The existing `scripts/evaluate.py` and `scripts/evaluate_qwen36.py` also accept released HF checkpoints through this interface. Training checkpoints and prepared feature caches retain their specialized evaluators.
