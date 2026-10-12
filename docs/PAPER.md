# Paper details

**VideoJev: System One Model for Video Understanding and Interaction**

VideoJev combines pretrained multimodal understanding with direct structured prediction. Temporal grounding and multiple-choice video QA share a candidate-scoring interface, providing reusable judgments for video understanding and interaction without autoregressive answer decoding.

## Authors and affiliations

Yuhan Zhu<sup>*,1</sup>, Xiangyu Zeng<sup>*,1</sup>, Qingyi Si<sup>*,‡,2</sup>, Chenxu Yang<sup>*,3</sup>, Yifei Li<sup>*,4</sup>, Junhao Zhou<sup>3</sup>, Dingyu Yao<sup>3</sup>, Jun Zhang<sup>2</sup>, Chuanyu Qin<sup>3</sup>, Luozheng Qin<sup>5</sup>, Zhiqiu Zhang<sup>6</sup>, Yuandong Yang<sup>1</sup>, Yuqi Xu<sup>2</sup>, Tianyu Zhang<sup>7</sup>, Hongcheng Guo<sup>8</sup>, Jiaqi Wang<sup>2</sup>, Nan Duan<sup>2</sup>, Limin Wang<sup>‡,†,1</sup>.

<sup>1</sup> NJU · <sup>2</sup> JD.com · <sup>3</sup> UCAS · <sup>4</sup> THU · <sup>5</sup> XMU · <sup>6</sup> SJTU · <sup>7</sup> USTC · <sup>8</sup> FDU

<sup>*</sup> Equal contribution · <sup>‡</sup> Project lead · <sup>†</sup> Corresponding author

## Method

![VideoJev architecture: unified candidates, shared prefix and scoring, and decision outputs](assets/videojev-method.png)

Temporal candidates identify native two-frame intervals; QA candidates contain the supplied answer text. Each candidate ends with `<|option_end|>`, and each question has a `<|decide|>` readout. Two learned projections map candidate and decision features into a normalized 512-dimensional space. Their cosine similarity, scaled by 10 with a learned bias, produces a shared candidate score.

Grounding applies sigmoid membership scores and decodes a contiguous event span. QA applies softmax over valid options and selects the highest-scoring answer. The pretrained causal backbone and native DeltaNet are retained, with the final decoder MLP removed. Vision and the visual merger stay frozen during training.

### Sharing video computation

| Paper inference mode | How it works |
|---|---|
| **Cached** | Encode and prefill the video once. Each arriving query branches independently from that prefix cache and needs one query-suffix forward. |
| **Joint** | Put multiple queries after one video prefix. A single causal language-model forward returns the scores for every question in the group. |

Neither mode inserts reference or predicted answers. In Joint inference, later questions can attend to earlier question text and candidates. Overlength groups split only between complete questions and repeat the complete video prefix.

The released `predict_video` and `predict_features` APIs implement Joint query grouping. `encode_video` / `predict_features` also allow visual feature reuse. They do not expose the paper's independent language-prefix Cached inference interface.

### Training

The paper uses TimeLens-100K and TimeLens2-93K for grounding, plus an internally curated corpus of approximately 240K multiple-choice questions over 19K videos. Head alignment uses 30% of videos; joint adaptation uses the remaining 70%. Each stage runs for one epoch, with questions from the same video sharing a prefix.

Head alignment uses BCE for grounding and CE for QA. Joint adaptation adds twice the duration-weighted SoftIoU loss to grounding; QA retains CE. The [data guide](DATA.md) describes the inputs required by the code release; these datasets are not bundled in this repository.

## Reported results and measurement protocol

The [README results](../README.md#results) display the manuscript's original main grounding and video QA tables, cropped directly from the compiled PDF. They are paper-reported measurements, not a new benchmark run of the BF16 Hugging Face exports.

| Setting | Temporal grounding | Video QA |
|---|---|---|
| Benchmarks | TimeLens-Bench refined Charades, ActivityNet, QVHighlights | LVBench test, Video-MME test, LongVideoBench validation, MLVU development MCQ |
| Quality metric | mIoU; the paper also reports R1@0.3/0.5/0.7 | Accuracy; MLVU uses seven-task M-Avg |
| Quality-evaluation context | 147456 tokens | 262144 tokens, fixed across benchmarks |
| Sampling | 4 FPS | 2 FPS, at most 2048 frames |
| Decoder | Threshold 0.25, smoothing 1, for both model sizes | Highest-scoring supplied option |
| Latency dataset | Full Charades: 1313 videos / 3363 queries | Full LVBench: 103 videos / 1549 questions |

Latency is model forward time per query, measured on H200 at batch size 1 after untimed warm-up. It includes visual encoding, prefill, and all autoregressive decoding steps for generative baselines; it excludes video decoding, preprocessing, data transfers, parsing, and metrics. VideoJev uses Joint inference. Generative baselines also share video encoding and prefix prefill, then answer from independent query branches. Grounding latency uses a 262144-token context for VideoJev, distinct from the 147456-token quality-evaluation context. Both tasks use frames only, without subtitles or audio.

Grounding speedups of 2.8× / 3.1× compare VideoJev-4B / 35B-A3B to their respective Instruct backbones. QA speedups of 129× / 100× compare them to their respective Thinking backbones. These ratios describe the measured workloads and forward-time protocol.

### Current release defaults

The existing code and model packages retain earlier evaluation defaults: grounding thresholds of 0.15 for 4B and 0.35 for 35B, and a QA workflow that evaluates both 196608 and 262144 contexts. These differ from the paper's fixed 0.25 threshold and 262144 QA context. The convenience API can also select a different raw-video resolution from prepared benchmark features. Use the manuscript settings and matching feature plans when reproducing the tables; running the default examples alone does not reproduce them.

The distributed 35B training export and the standalone Hugging Face model package also serve different loading paths; see [Checkpoint export](CHECKPOINTS.md). The standalone packages store BF16 weights, while the paper's 4B QA measurements used FP32 weights with BF16 autocast. Export validation checks tensor conversion and reference inference under the same casting policy; full benchmark metrics have not been rerun for these BF16 packages.

## Source notes

The title, author order, and affiliations follow the manuscript's `main.tex`. The method and inference descriptions follow `sections/03_method.tex`; the results follow Tables 1–2 in `sections/04_experiments.tex`; measurement settings follow `sections/99_appendix.tex`.

The README teaser and method image are copied from the manuscript's `figures/video-jev-teaser/video-jev-teaser.png` and `figures/video-jev-charades-v48/video-jev-charades-v48.png`. The method figure's scores and example interval are illustrative.

The results images `temporal-grounding-table.png` and `video-qa-table.png` reproduce Tables 1 and 2 from `main.pdf`, including their captions, complete model comparisons, and original formatting.
