# Paper details

**VideoJev: System One Model for Video Understanding and Interaction**

## Authors and affiliations

Yuhan Zhu<sup>*,1</sup>, Xiangyu Zeng<sup>*,1</sup>, Qingyi Si<sup>*,‡,2</sup>, Chenxu Yang<sup>*,3</sup>, Yifei Li<sup>*,4</sup>, Junhao Zhou<sup>3</sup>, Dingyu Yao<sup>3</sup>, Jun Zhang<sup>2</sup>, Chuanyu Qin<sup>3</sup>, Luozheng Qin<sup>5</sup>, Zhiqiu Zhang<sup>6</sup>, Yuandong Yang<sup>1</sup>, Yuqi Xu<sup>2</sup>, Tianyu Zhang<sup>7</sup>, Hongcheng Guo<sup>8</sup>, Jiaqi Wang<sup>2</sup>, Nan Duan<sup>2</sup>, Limin Wang<sup>‡,†,1</sup>.

<sup>1</sup> NJU · <sup>2</sup> JD.com · <sup>3</sup> UCAS · <sup>4</sup> THU · <sup>5</sup> XMU · <sup>6</sup> SJTU · <sup>7</sup> USTC · <sup>8</sup> FDU

<sup>*</sup> Equal contribution · <sup>‡</sup> Project lead · <sup>†</sup> Corresponding author

## Method

![VideoJev architecture](assets/videojev-method.png)

A shared 512-dimensional Q/K head scores temporal intervals and QA options by cosine similarity (scale 10 plus learned bias). Grounding uses sigmoid scores; QA uses softmax. The causal Qwen backbone and DeltaNet remain, with the final decoder MLP removed and vision frozen.

| Mode | Video computation |
|---|---|
| Cached | Reuse a video prefix cache; each arriving query runs independently. |
| Joint | Score multiple questions after one video prefix in a single causal forward. |

Released APIs support Joint grouping and visual feature reuse. Groups split between complete questions when they exceed the context limit.

### Training

Grounding uses TimeLens-100K and TimeLens2-93K; QA uses roughly 240K questions over 19K videos. Head alignment uses 30% of videos and joint adaptation 70%, one epoch per stage. See the [training recipe](../README.md#training) and [data requirements](DATA.md).

## Reported results and measurement protocol

[Original paper tables](../README.md#results).

| Setting | Temporal grounding | Video QA |
|---|---|---|
| Benchmarks | TimeLens-Bench Charades, ActivityNet, QVHighlights | LVBench, Video-MME, LongVideoBench, MLVU |
| Metric | mIoU, R1@0.3/0.5/0.7 | Accuracy; M-Avg for MLVU |
| Quality context | 147456 tokens | 262144 tokens |
| Sampling | 4 FPS | 2 FPS, at most 2048 frames |
| Decoder | Threshold 0.25, smoothing 1 | Highest-scoring option |
| Latency dataset | Full Charades | Full LVBench |

Latency per query on H200 (batch 1, after warm-up) includes visual encoding and language-model prefill/decoding; video decoding, preprocessing, transfers, parsing and metrics are excluded. Generative baselines share video encoding and prefix prefill. VideoJev uses Joint inference and a 262144-token grounding context. Both tasks use frames only.

### Current release defaults

- Grounding defaults to `threshold=0.25`, with per-call overrides.
- QA scripts compare two contexts; the paper fixes 262144. Raw-video resolution may differ from prepared benchmark features.
- HF exports use BF16. Paper 4B QA used FP32 weights with BF16 autocast. Full benchmarks have not been rerun on the HF exports.
