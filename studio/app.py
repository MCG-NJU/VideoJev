"""VideoJev's public Gradio demo for temporal grounding and video QA."""

from __future__ import annotations

import html
import os
import re
import time
from pathlib import Path

import gradio as gr

from inference import predict

ROOT = Path(__file__).resolve().parent
EXAMPLE_VIDEO = ROOT / 'examples/eating_spaghetti.mp4'
MAX_DURATION = 30.0
DEFAULT_OPTIONS = 'A. Pasta.\nB. Rice.\nC. Soup.'

CLAY = gr.themes.colors.Color(
    c50='#fdf7f5', c100='#f9ede9', c200='#f2d8d0', c300='#e9b8ab',
    c400='#d9917e', c500='#c77469', c600='#b7483d', c700='#92362f',
    c800='#712b27', c900='#55221f', c950='#341613', name='clay',
)

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#vj-header { margin: 5px 0 22px; }
#vj-header h1 { font: 500 44px/1.1 Georgia,serif; letter-spacing: -.05em; margin: 0 0 7px; color: #292824; }
#vj-header h1 em { color: #b7483d; font-weight: 400; }
#vj-header p { color: #65615d; font-size: 15px; margin: 0; }
#vj-header a { color: #a3453b; text-decoration: none; }
#vj-header a:hover { text-decoration: underline; }
#vj-header .vj-mark { display: inline-block; width: 38px; height: 38px; vertical-align: -.12em; margin-right: 7px; }
#vj-inputs { border: 1px solid #e9e4dc; border-radius: 14px; padding: 12px; background: #fffcf8; }
#vj-run { background: #b7483d !important; border-color: #b7483d !important; }
#vj-run:hover { background: #96392f !important; }
.vj-results { display: grid; gap: 14px; margin-top: 10px; }
.vj-result { border: 1px solid #e9e4dc; border-radius: 12px; padding: 17px 20px; background: #fffcf8; }
.vj-kicker { color: #a3453b; text-transform: uppercase; letter-spacing: .11em; font: 700 11px Arial,sans-serif; margin-bottom: 8px; }
.vj-answer { color: #292824; font: 500 25px/1.25 Georgia,serif; margin-bottom: 10px; }
.vj-query { color: #77716a; font-size: 13px; margin-bottom: 12px; }
.vj-timebar { position: relative; width: 100%; height: 10px; border-radius: 8px; background: #eee6df; overflow: hidden; }
.vj-timebar span { position: absolute; top: 0; bottom: 0; background: #b7483d; border-radius: 8px; }
.vj-range { display: flex; justify-content: space-between; color: #77716a; font-size: 12px; margin-top: 5px; }
.vj-option { display: grid; grid-template-columns: minmax(0, 1fr) 140px 44px; align-items: center; gap: 12px; margin-top: 9px; font-size: 13px; }
.vj-option.is-best { font-weight: 700; }
.vj-score { height: 8px; background: #eee6df; border-radius: 8px; overflow: hidden; }
.vj-score span { display: block; height: 100%; background: #b9a89e; border-radius: 8px; }
.vj-option.is-best .vj-score span { background: #b7483d; }
.vj-footnote { color: #77716a; font-size: 12px; margin: 4px 2px 0; }
@media (max-width: 600px) {
  #vj-header h1 { font-size: 36px; }
  .vj-option { grid-template-columns: minmax(0, 1fr) 90px 40px; gap: 8px; }
}
.gradio-container footer { display: none !important; }
"""

HEADER = """
<div id="vj-header">
  <h1><svg class="vj-mark" viewBox="0 0 104 104" aria-hidden="true">
    <g transform="translate(4 8)">
      <path fill="#292824" d="M5 0H27L48 56L36 88Z"/>
      <path fill="#D6A19A" d="M70 0H92L81.5 28H59.5Z"/>
      <path fill="#C77469" d="M57.5 33H79.5L70.1 58H48.1Z"/>
      <path fill="#B7483D" d="M46.2 63H68.2L59 88H36Z"/>
    </g>
  </svg>Video<em>Jev</em></h1>
  <p>One video, two kinds of decisions. Try temporal grounding and multiple-choice video QA.
     <a href="https://arxiv.org/abs/2610.18615" target="_blank">Paper ↗</a> ·
     <a href="https://github.com/MCG-NJU/VideoJev" target="_blank">Code ↗</a></p>
</div>
"""


def video_duration(path: str) -> float:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / 1_000_000)
    raise ValueError('Could not read the video duration.')


def build_questions(grounding: str, qa: str, options_text: str):
    grounding = (grounding or '').strip()
    qa = (qa or '').strip()
    questions = []
    if grounding:
        questions.append({'id': 'grounding', 'task': 'grounding', 'query': grounding})
    if qa:
        raw = [line.strip() for line in (options_text or '').splitlines() if line.strip()]
        if not 2 <= len(raw) <= 8:
            raise ValueError('Video QA needs 2–8 answer options, one per line.')
        options = []
        for index, option in enumerate(raw):
            if not re.match(r'^[A-H][.)]\s+', option):
                option = f'{chr(65 + index)}. {option}'
            options.append(option)
        questions.append({'id': 'qa', 'task': 'choice', 'query': qa, 'options': options})
    if not questions:
        raise ValueError('Enter a grounding query, a video question, or both.')
    return questions


def render_results(results, questions, duration: float, elapsed: float) -> str:
    by_id = {question['id']: question for question in questions}
    cards = []
    for result in results:
        question = by_id[result['id']]
        query = html.escape(question['query'])
        if result['task'] == 'grounding':
            start, end = result['interval']
            left = max(0.0, min(100.0, start / duration * 100))
            right = max(left, min(100.0, end / duration * 100))
            cards.append(f'''<div class="vj-result">
              <div class="vj-kicker">Temporal grounding</div>
              <div class="vj-answer">{start:.1f}–{end:.1f} seconds</div>
              <div class="vj-query">{query}</div>
              <div class="vj-timebar"><span style="left:{left:.3f}%;width:{right-left:.3f}%"></span></div>
              <div class="vj-range"><span>0 s</span><span>{duration:.1f} s</span></div>
            </div>''')
        else:
            answer = html.escape(result['answer'])
            options = question['options']
            bars = []
            for index, (option, probability) in enumerate(zip(options, result['probabilities'])):
                label = html.escape(option)
                best = ' is-best' if index == result['answer_index'] else ''
                width = max(0.0, min(100.0, float(probability) * 100))
                bars.append(f'''<div class="vj-option{best}"><span>{label}</span>
                  <span class="vj-score"><span style="width:{width:.2f}%"></span></span>
                  <span>{width:.1f}%</span></div>''')
            cards.append(f'''<div class="vj-result">
              <div class="vj-kicker">Video question answering</div>
              <div class="vj-answer">{answer}</div>
              <div class="vj-query">{query}</div>
              {''.join(bars)}
            </div>''')
    return '<div class="vj-results">' + ''.join(cards) + f'<p class="vj-footnote">Processed in {elapsed:.1f} s · Probabilities below</p></div>'


def run(video, grounding, qa, options, threshold):
    if not video:
        raise gr.Error('Upload a video first.')
    try:
        questions = build_questions(grounding, qa, options)
        duration = video_duration(video)
        if duration <= 0 or duration > MAX_DURATION:
            raise ValueError(f'Use a video up to {MAX_DURATION:.0f} seconds long.')
        context = 8192 if duration <= 15 else 16384
        start = time.perf_counter()
        results = predict(video, questions, threshold=float(threshold), context=context)
        public_results = [{key: value for key, value in result.items() if key != 'logits'} for result in results]
        return render_results(results, questions, duration, time.perf_counter() - start), public_results
    except ValueError as error:
        raise gr.Error(str(error)) from error
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            raise gr.Error('GPU memory is full. Try a shorter video.') from error
        raise gr.Error(str(error)) from error


def build_demo():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=CLAY, secondary_hue=CLAY), css=CSS, title='VideoJev') as demo:
        gr.HTML(HEADER)
        with gr.Row(elem_id='vj-inputs'):
            with gr.Column(scale=6):
                video = gr.Video(label='Video · up to 30 seconds', value=str(EXAMPLE_VIDEO), sources=['upload'])
            with gr.Column(scale=5):
                grounding = gr.Textbox(label='Find this event', value='The woman in green eats pasta.', lines=2)
                qa = gr.Textbox(label='Ask about the video', value='What is the woman in green eating?', lines=2)
                options = gr.Textbox(label='Answer options · one per line', value=DEFAULT_OPTIONS, lines=3)
                with gr.Accordion('Grounding settings', open=False):
                    threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label='Grounding threshold')
                button = gr.Button('Analyze video', variant='primary', elem_id='vj-run')
        answer = gr.HTML(label='Result')
        with gr.Accordion('Prediction data', open=False):
            probabilities = gr.JSON(label='Probabilities')
        button.click(run, [video, grounding, qa, options, threshold], [answer, probabilities], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1)
    return demo


demo = build_demo()

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')))
