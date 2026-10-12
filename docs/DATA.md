# Data preparation

Supply prepared datasets under `data/`. Paths in `configs/*.json` resolve from the repository root; absolute paths also work.

```text
data/
  ready.json, split.json
  head_only/
    train.jsonl, episodes.json, packs.json
    vtg0.jsonl, vtg0_lengths.json
    vtg1.jsonl, vtg1_lengths.json
    qa.jsonl, qa_lengths.json
  joint/                     # same layout as head_only
  metadata/                  # media geometry
  qa/                        # option permutation metadata
  cache/                     # visual features
  benchmarks/<name>/
    questions.jsonl
    official.json            # LVBench uses official.jsonl
    lengths_f2.json          # QA
    lengths_f4.json          # grounding
```

## Manifests and media

Every row needs `id`, `video_key`, `video_path`, `duration` and `query`:

- Grounding: `task="grounding"`, `spans` containing all target intervals.
- QA: `task="choice"`, labeled `options`, zero-based `answer_index`.

Keep IDs unique and questions in source order. [QA conversion helpers](../timejev/qa_manifest.py) preserve the option format.

Optional `media_source` selects `video`, `frames_sidecar` or `frames_4fps` by `kind` and `path`. QA context planning needs complete 2-FPS frames with `.complete.json` and `.timestamps.jsonl` sidecars, usually under `data/frames/<benchmark>/<video-stem>/`.

## Features and plans

- Feature caches contain BF16 native visual features, grid/timestamp metadata and model/file hashes. Use features from the matching model.
- `episodes.json` binds manifests, caches, context, FPS and seed; each episode records ordered `query_indices` and exact `tokens`.
- `packs.json` binds episode metadata and packer hashes. `ready.json` records completed data files and their hashes.

See [episode loading](../timejev/multiturn_data.py), [packing](../timejev/packing.py) and [feature formats](../timejev/features.py).

RGB planners reuse the manifests and video split, then regenerate lengths and packs:

| Planner | Output |
|---|---|
| `scripts/prepare_rgb192.py` | `data/rgb192/` |
| `scripts/prepare_qwen36_mixed.py` | `data/qwen36_mixed_v1/` |
| `scripts/prepare_context_eval.py` | QA context plans |

Run each with `--help`. Rebuild metadata after changing media paths, tokenizer, context or input code.

The RGB planners require 404037 training questions: 96586 TimeLens-100K, 67924 TimeLens2 and 239527 QA.
