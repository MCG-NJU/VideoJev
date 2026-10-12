# Input data contract

The training/evaluation code consumes a prepared data bundle. Download the original datasets under their respective terms; this repository contains no annotations or media. The previous experiment-directory importer is intentionally not part of the public code.

The feature recipe expects this layout (paths may be changed in the JSON configurations):

```text
data/
  ready.json
  split.json
  head_only/                 # first-stage video split
    train.jsonl
    episodes.json
    packs.json
    vtg0.jsonl, vtg0_lengths.json
    vtg1.jsonl, vtg1_lengths.json
    qa.jsonl, qa_lengths.json
  joint/                     # same schema for the joint-stage split
  metadata/                  # original media/feature geometry metadata
  qa/                        # option permutation metadata
  cache/                     # frozen native merger features and provenance
  benchmarks/<name>/
    questions.jsonl
    official.json            # QA references; LVBench uses official.jsonl
    lengths_f2.json          # QA feature lengths
    lengths_f4.json          # grounding feature lengths
```

Each manifest row has `id`, `video_key`, `video_path`, `duration`, and `query`. Grounding rows contain `task: "grounding"` and all GT intervals in `spans`. QA rows contain `task: "choice"`, the original labeled `options`, and a zero-based `answer_index`. Keep question IDs unique and source-record question order intact. `timejev/qa_manifest.py` provides the QA conversion helpers.

An optional `media_source` object selects `kind: "video"`, `"frames_sidecar"`, or `"frames_4fps"` with an explicit `path`. QA context planning requires complete 2-FPS frame extractions with `.complete.json` and `.timestamps.jsonl` sidecars; conventional locations are `data/frames/<benchmark>/<video-stem>/`. RGB geometry, sampling and sidecar validation are implemented in `timejev/rgb_media.py` and `timejev/context_eval.py`.

Feature caches store BF16 native visual-merger outputs, grid/timestamp metadata, a model identity, and content hashes. Do not mix features extracted by different models. `timejev/features.py`, `timejev/adaptive_features.py`, and `timejev/feature_catalog.py` define the cache formats and validation. Full-length training videos are never truncated.

`episodes.json` binds the manifest SHA256, input implementation SHA256, source/cache bindings, context, FPS and seed, and includes each episode's ordered `query_indices` and exact `tokens`. `packs.json` binds the episode metadata hash and XTuner packer source hashes. See `timejev/multiturn_data.py:load_episode_spec` and `timejev/packing.py` for exact contracts. `ready.json` is a completion inventory with hashes of the prepared data files; do not create a dummy completion marker.

The RGB planners reuse `head_only/train.jsonl`, `joint/train.jsonl`, their video membership metadata, and `split.json`, then regenerate exact lengths and packing for the selected model. Their outputs are `data/rgb192/` or `data/qwen36_mixed_v1/`; the latter includes `evaluation_147456.json`, `evaluation_196608.json` and `evaluation_262144.json`.

The fixed training inventory has 404037 questions: 96586 TimeLens-100K, 67924 TimeLens2, and 239527 QA questions. Changing this inventory or a path-bound source requires rebuilding and auditing the metadata, rather than reusing old hashes.
