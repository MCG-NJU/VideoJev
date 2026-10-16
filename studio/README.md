---
title: VideoJev
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
license: apache-2.0
models:
  - MCG-NJU/VideoJev-4B
deployspec:
  entry_file: app.py
---

# VideoJev interactive demo

Try temporal grounding and video QA with [VideoJev-4B](https://modelscope.cn/models/MCG-NJU/VideoJev-4B). Upload a video up to 30 seconds or use the sample, then ask one or both questions. The app shows time spans, answers, and probabilities.
