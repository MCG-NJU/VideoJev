# Evaluation protocol

This guide describes the current code entrypoints and their existing defaults. For the manuscript's fixed grounding threshold and QA context, see [Paper details](PAPER.md#reported-results-and-measurement-protocol).

The published entrypoints preserve `episode_lengths` and `turn_episode_indices`: one complete video prefix, followed by its questions and options in manifest order, with one native forward per query group. No answers or predicted answers are inserted. Greedy groups respect the 64-token-aligned context bound and reset all language state between groups. A single question that cannot fit causes an explicit error.

| Benchmark | Questions | Main metric |
|---|---:|---|
| Charades | 3363 | mIoU and R1@0.3/0.5/0.7 |
| ActivityNet | 4500 | mIoU and R1@0.3/0.5/0.7 |
| QVHighlights | 1541 | mIoU and R1@0.3/0.5/0.7 |
| LVBench | 1549 | Accuracy |
| Video-MME | 2700 | Accuracy |
| LongVideoBench validation | 1337 | Accuracy |
| MLVU Dev multiple-choice tasks | 2174 | M-Avg |

Grounding uses 4 FPS and the original feature geometry. The final 4B protocol uses 147456 tokens and a fixed decoder (threshold 0.15, smoothing 1, `round_repair_clamp_v1`). The RGB 192K training entrypoint has its own matching 192K evaluation geometry.

Final QA evaluation uses both 196608 and 262144 contexts, 2 FPS, at most 2048 frames, and no subtitles/audio. Report the larger main metric for each benchmark, with ties resolved to 196608; use the same selected prediction set for any subsequent analysis. Context-level files retain the evidence needed to reproduce the selection. The seven-benchmark feature launcher also provides evaluation at its training context; use the two-context evaluator for final QA reporting.

`evaluate_4b_multiturn.py` accepts an input-only plan with all eleven scopes, ordered query membership, immutable input-file hashes, feature paths/checksums, decoder parameters and the cache's model/completion identities. `prepare_multiturn_eval.py --reference /path/to/input_bundle --out /path/to/plan.json` freezes an existing verified bundle containing `identity.json`, `plan.json`, `decoder.json` and `inputs/`; it copies no checkpoint or predictions.

Every evaluator checks complete query coverage and refuses mixed identities when resuming. Use a new output directory after changing code, weights, inputs or protocol. These checks are not a substitute for a full benchmark run.
