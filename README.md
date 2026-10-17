# VideoJev project homepage

A responsive, dependency-free English project page for **VideoJev: System One Model for Video Understanding and Interaction**.

## Preview

Run from this directory:

```bash
python serve.py --port 8000
```

Open `http://127.0.0.1:8000`. The preview server binds only to localhost and supports byte-range requests so that clicking a frame can seek through the videos. The site also works directly from `index.html`; it does not fetch data or require a build step. All assets and fonts are local.

## Structure

- `index.html`: page sections, author metadata, citation, and release states.
- `styles.css`: responsive layout and paper-derived warm neutral/terracotta palette.
- `app.js`: recorded examples, inference diagrams, figure viewer, table filters and copy citation.
- `assets/data/paper-data.js`: browser-ready data, including recorded predictions and Tables 1–2.
- `assets/data/paper-data.json`: the same data in reusable JSON format.
- `assets/figures/`: figures exported from the manuscript's vector PDFs.
- `assets/frames/`, `assets/videos/`: real benchmark frames and silent display video.
- `assets/VideoJev.pdf`: manuscript snapshot used for this page.
- `prepare_assets.py`: asset export script; set `ROOT` to the paper workspace before rerunning. Requires PyMuPDF, Pillow, fontTools, Brotli and FFmpeg.

## Scientific scope

The page uses the current manuscript and its LaTeX tables. Table values are parsed directly; website formatting does not recompute rounded averages. Website generation does not change paper sources or run inference.

Examples replay selected saved benchmark predictions. Case 01 plays the complete 27.3-second Charades video and shows three of six query readouts. Case 04 plays a 48-second excerpt while the recorded model input was the complete 3,665.5-second video, with 18 questions; two readouts are displayed. Display frames are illustrative video evidence, not a claim about the exact frame sampling used during evaluation. Option bars show uncalibrated scores.

Main latency is amortized **model forward time**: shared visual encoding, prefill and decoding are included; video decoding, preprocessing, data transfers and output postprocessing are excluded. Baselines also share the prefix. The 100–129× QA comparison is with the corresponding Thinking models. Figure 3 is a separate 40-video-per-task query-scaling experiment; its sampling, timing and independent quality scores must not be mixed with Tables 1–2.

Table 3 is a same-data single-stage training control, not the default two-stage checkpoint. Table 4's 35B Cached and Joint runs differ in language-weight precision; the page retains this qualification.

## Release information

[Paper · arXiv:2610.18615](https://arxiv.org/abs/2610.18615).

## Hosting

Serve this folder as a static website, including GitHub Pages. Paths are relative, so deployment under a project subdirectory works. `.nojekyll` bypasses Jekyll. No server-side runtime, remote font service, analytics or inference endpoint is required.

## Source and credits

Figures, author list, benchmark results and cases: VideoJev manuscript and accompanying research assets. Page organization is inspired by the OneStreamer project page, with an original implementation and paper-matched visual design. Inter fonts retain their bundled SIL Open Font License in `assets/fonts/OFL.txt`. The header, navigation and footer share the approved VideoJev vector wordmark; the favicon uses its segmented V. The same wordmark appears in the downloadable paper title.

## Editorial presentation

The page stands alone: repeated figure/table numbers, source links and usage captions are omitted. The two-paragraph model overview and optional full abstract sit below the resource links and above the key results; the earlier hero slogan and separate overview section are removed. Essential metric definitions stay with the results, and evaluation conditions are collected in one expandable section. Detailed source provenance remains documented here and in the data export script. Comparison tables are displayed directly below their analysis text, without disclosure controls or duplicate numeric summaries. Table values and the inference-precision qualification are retained.

## Heading typography

Section headlines use a shared Candara bold italic style and a 36px desktop size. Explicit line breaks are removed, and headings span the section width where needed. On narrower screens, a shared size fits every headline on one line. All headline text, including emphasis, uses warm dark gray (#55534f), slightly darker than the body copy, and shares the same typography. The two analysis card headings are subordinate to “What makes the difference?”: they use 24px on desktop and scale down to 16px on narrow screens, while retaining the same bold italic Candara style. A smaller gap below the parent heading and lighter card dividers reinforce the grouping.

Native Candara is preferred. The local WOFF2 fallback contains Candara Regular, with bold and italic synthesized by the browser; it is not a native Candara Italic font. Font origin and original copyright/license metadata are retained in `assets/fonts/CANDARA-NOTICE.txt` and the font file.

## Citation

```bibtex
@article{zhu2026videojev,
  title={VideoJev: System One Model for Video Understanding and Interaction},
  author={Zhu, Yuhan and Zeng, Xiangyu and Si, Qingyi and Yang, Chenxu and Li, Yifei and Qin, Luozheng and Zhou, Junhao and Yao, Dingyu and Zhang, Jun and Qin, Chuanyu and others},
  journal={arXiv preprint arXiv:2610.18615},
  year={2026}
}
```
