'use strict';

(() => {
  const data = window.PAPER_DATA;
  if (!data) return;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escape = (value) => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const time = value => `${Math.floor(value / 60).toString().padStart(2,'0')}:${Math.floor(value % 60).toString().padStart(2,'0')}`;
  const seconds = value => Number.isInteger(value) ? String(value) : value.toFixed(1);
  const probability = value => value > .999 && value < 1 ? '>99.9%' : value > 0 && value < .001 ? '<0.1%' : `${(value * 100).toFixed(1)}%`;

  let demoId = 1;
  let demoModel = '4B';
  let queryIndex = 0;
  const video = $('#demo-video');
  const promo = $('#promo-video');
  if (promo) {
    const promoPlay = document.createElement('button');
    promoPlay.className = 'promo-play';
    promoPlay.setAttribute('aria-label', 'Play the 90-second VideoJev overview');
    promoPlay.innerHTML = '<span aria-hidden="true">▶</span>';
    promo.parentElement.append(promoPlay);
    promo.controls = false;
    promoPlay.addEventListener('click', async () => {
      promoPlay.hidden = true;
      promo.controls = true;
      try { await promo.play(); }
      catch { promoPlay.hidden = false; promo.controls = false; }
    });
    promo.addEventListener('play', () => video.pause());
    video.addEventListener('play', () => promo.pause());
  }
  const playButton = document.createElement('button');
  playButton.className = 'video-play';
  playButton.setAttribute('aria-label', 'Play the example video');
  playButton.innerHTML = '<span class="play-circle" aria-hidden="true">▶</span>';
  $('.video-wrap').append(playButton);
  video.controls = false;
  playButton.addEventListener('click', async () => {
    video.controls = true;
    playButton.hidden = true;
    try { await video.play(); }
    catch { playButton.hidden = false; video.controls = false; }
  });
  const currentCase = () => data.cases.find(c => c.id === demoId);
  let pendingSeek = null;
  video.addEventListener('loadedmetadata', () => {
    if (pendingSeek !== null) { video.currentTime = pendingSeek; pendingSeek = null; }
  });
  function seek(value) {
    video.controls = true;
    playButton.hidden = true;
    if (video.readyState >= 1) video.currentTime = value;
    else {
      pendingSeek = value;
      video.preload = 'metadata';
      video.load();
    }
  }
  function showActiveFrame(value) {
    const frames = currentCase().frames;
    const nearest = frames.reduce((best, f, i) => Math.abs(f.time-value) < Math.abs(frames[best].time-value) ? i : best, 0);
    $$('.frame-button').forEach((button, i) => {
      button.classList.toggle('active', i === nearest);
      button.setAttribute('aria-pressed', String(i === nearest));
    });
  }
  video.addEventListener('timeupdate', () => showActiveFrame(video.currentTime));

  function renderPrediction() {
    const c = currentCase();
    const q = c.queries[queryIndex];
    const pred = q.predictions[demoModel];
    $$('.query-button').forEach((button, i) => {
      button.classList.toggle('active', i === queryIndex);
      button.setAttribute('aria-pressed', String(i === queryIndex));
    });
    if (c.id === 1) {
      const [start,end] = pred.prediction;
      const [gtStart,gtEnd] = q.spans[0];
      const intersection = Math.max(0, Math.min(end,gtEnd)-Math.max(start,gtStart));
      const union = Math.max(end,gtEnd)-Math.min(start,gtStart);
      const row = (label, a, b, cls) => `<div class="interval-row ${cls}"><span>${label}</span><div class="interval-track"><span class="interval-bar" style="left:${a/c.duration*100}%;width:${(b-a)/c.duration*100}%"></span></div><span>${seconds(a)}–${seconds(b)} s</span></div>`;
      $('#prediction-detail').innerHTML = `<div class="prediction-heading"><h4>Temporal alignment</h4><span class="iou-score">IoU ${(intersection/union).toFixed(3)}</span></div>${row('Ground truth',gtStart,gtEnd,'gt')}${row('VideoJev',start,end,'ours')}<div class="interval-axis"><span>0 s</span><span>${c.duration.toFixed(1)} s</span></div>`;
    } else {
      const optionRows = q.options.map((option,i) => `<div class="option-result ${i===pred.answer_index?'selected':''}"><span class="score-fill" style="width:${pred.probabilities[i]*100}%"></span><span>${escape(option)}</span><span class="option-value">${escape(probability(pred.probabilities[i]))}${i===pred.answer_index?' ✓':''}</span></div>`).join('');
      $('#prediction-detail').innerHTML = `<div class="prediction-heading"><h4>Option scores</h4><span class="answer-key">Correct answer: ${String.fromCharCode(65+q.answer)}</span></div>${optionRows}`;
    }
  }

  function renderDemo(loadVideo = true) {
    const c = currentCase();
    const isGrounding = c.id === 1;
    $$('.demo-tab').forEach(tab => {
      const active = Number(tab.dataset.demo) === demoId;
      tab.classList.toggle('active',active);
      tab.setAttribute('aria-selected',String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    $('#demo-panel').setAttribute('aria-labelledby',isGrounding?'tab-grounding':'tab-qa');
    if (loadVideo) {
      video.pause();
      video.controls = false;
      playButton.hidden = false;
      pendingSeek = null;
      video.poster = c.frames[isGrounding ? 3 : 2].src;
      video.setAttribute('aria-label',isGrounding?'Full Charades video used for temporal grounding':'Opening excerpt from the hour-long LVBench video');
      $('source',video).src = `assets/videos/case-${String(c.id).padStart(2,'0')}.mp4`;
      video.load();
    }
    $('#video-caption').textContent = isGrounding ? '27-second video' : '48-second excerpt · full model input: 61:05';
    $('#query-count').textContent = `${c.queries.length} of ${c.groupSize} queries shown`;
    $('#flow-output').textContent = `${c.groupSize} decisions`;
    $('#frame-strip').innerHTML = c.frames.map((frame,i) => `<button class="frame-button ${i===0?'active':''}" data-frame="${i}" aria-label="Seek to ${time(frame.time)}" aria-pressed="${i===0}"><img src="${frame.src}" alt="Video frame at ${time(frame.time)}" width="160" height="100"><span>${time(frame.time)}</span></button>`).join('');
    $$('.frame-button').forEach(button => button.addEventListener('click',() => {
      const frame = c.frames[Number(button.dataset.frame)];
      seek(frame.time);
      showActiveFrame(frame.time);
    }));
    $('#query-buttons').innerHTML = c.queries.map((q,i) => `<button class="query-button ${i===queryIndex?'active':''}" data-query="${i}" aria-pressed="${i===queryIndex}"><span class="query-index">Q${i+1}</span><span>${escape(q.query)}</span><span class="query-arrow" aria-hidden="true">↗</span></button>`).join('');
    $$('.query-button').forEach(button => button.addEventListener('click',() => {
      queryIndex = Number(button.dataset.query);
      renderPrediction();
      seek(isGrounding ? c.queries[queryIndex].predictions[demoModel].prediction[0] : queryIndex === 0 ? 18 : 38);
    }));
    renderPrediction();
    showActiveFrame(c.frames[isGrounding ? 3 : 2].time);
  }
  $$('.demo-tab').forEach(tab => tab.addEventListener('click',() => {
    if (Number(tab.dataset.demo) === demoId) return;
    demoId = Number(tab.dataset.demo);
    queryIndex = 0;
    renderDemo();
  }));
  $('.demo-tabs').addEventListener('keydown',event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = $$('.demo-tab');
    const next = event.key==='Home'?tabs[0]:event.key==='End'?tabs[1]:tabs.find(t => t.getAttribute('aria-selected')==='false');
    next.click();
    next.focus();
  });
  $$('[data-demo-model]').forEach(button => button.addEventListener('click',() => {
    demoModel = button.dataset.demoModel;
    $$('[data-demo-model]').forEach(b => {b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    renderPrediction();
  }));
  renderDemo(false);

  const extraTitles = {6:'Count recurring events.',7:'See where models disagree.',8:'Inspect a shared failure.'};
  $('#extra-cases').innerHTML = [6,7,8].map(id => {
    const c = data.cases.find(c => c.id === id);
    const q = c.queries[0];
    const answers = ['4B','35B-A3B'].map(model => `<span>${model}<b>${escape(q.options[q.predictions[model].answer_index])}</b></span>`).join('');
    return `<article class="extra-case"><img src="${c.frames[id===8?4:0].src}" loading="lazy" alt="${escape(c.title)}"><div class="extra-case-body"><h4>${extraTitles[id]}</h4><p>${escape(q.query)}</p><div class="extra-answers">${answers}<span>Ground truth<b>${escape(q.options[q.answer])}</b></span></div></div></article>`;
  }).join('');

  function renderInference(mode) {
    const joint = mode === 'joint';
    $('#inference-description').textContent = joint
      ? 'Answer a group of questions in one forward pass.'
      : 'Reuse the video cache as new questions arrive, with one forward pass per query.';
    const label = '<div class="diagram-label"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 3h14v14H3zM7 6l7 4-7 4z"/></svg> Video encoded once</div>';
    $('#inference-diagram').innerHTML = joint ? `${label}<div class="sequence"><span class="token-video">V</span><span class="sequence-arrow">→</span><span class="token-query">Q<sub>1</sub> + options</span><span class="token-query">Q<sub>2</sub> + options</span><span class="token-query">Q<sub>3</sub> + options</span></div><div class="group-bracket"></div><div class="group-output">One forward <strong>A<sub>1</sub> · A<sub>2</sub> · A<sub>3</sub></strong></div>`
      : `${label}<div class="cache-branches"><div class="cache-source">Video cache<span>prefill once</span></div><div class="branches">${[1,2,3].map(n=>`<div class="branch"><span>↗</span><span>Q<sub>${n}</sub> + options</span><small>→ A<sub>${n}</sub></small></div>`).join('')}</div></div>`;
    $$('[data-inference]').forEach(b => {const selected=b.dataset.inference===mode;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
  }
  $$('[data-inference]').forEach(button => button.addEventListener('click',()=>renderInference(button.dataset.inference)));
  renderInference('joint');

  $$('[data-scaling]').forEach(button => button.addEventListener('click',() => {
    const ground = button.dataset.scaling === 'grounding';
    $$('[data-scaling]').forEach(b => {b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    $('#scaling-image').src = `assets/figures/scaling-${ground?'grounding':'qa'}.webp`;
    $('#scaling-image').alt = ground ? 'Multi-query grounding time: VideoJev Joint takes 0.39 seconds for ten queries.' : 'Multi-query QA time: VideoJev Joint takes 1.64 seconds for ten queries, compared with 1.67 seconds for Direct Joint.';
    $('#scaling-zoom').dataset.zoom = $('#scaling-image').getAttribute('src');
    $('#scaling-value').innerHTML = `${ground?'0.39':'1.64'}<span>s</span>`;
    $('#scaling-description').innerHTML = ground ? '10 grounding queries<br> <strong>vs. 5.98 s</strong> for Instruct' : '10 QA queries<br> <strong>vs. 1.67 s</strong> for Direct Joint';
    $('#scaling-insight').hidden = ground;
    $('#scaling-insight').textContent = 'LVBench accuracy: 67.9% vs. 59.8% for Direct Joint.';
  }));

  let tableTask = 'grounding';
  function renderTable() {
    const ground = tableTask === 'grounding';
    const full = $('#show-all-methods').checked;
    const metrics = $('#show-all-metrics').checked && ground;
    const scale = $('#table-scale').value;
    $('#all-metrics-label').hidden = !ground;
    const selected = ground ? ['VideoJev-4B','VideoJev-35B-A3B','Qwen3.5-4B','Qwen3.6-35B-A3B','TimeLens2-4B'] : ['VideoJev-4B','VideoJev-35B-A3B','Qwen3.5-4B','Qwen3.6-35B-A3B'];
    const rows = data.tables[tableTask].filter(row => (full || selected.includes(row.model)) && (scale==='all'||row.size===scale));
    const headings = ground
      ? metrics ? ['Charades<br>R1@.3','Charades<br>R1@.5','Charades<br>R1@.7','Charades<br>mIoU','ActivityNet<br>R1@.3','ActivityNet<br>R1@.5','ActivityNet<br>R1@.7','ActivityNet<br>mIoU','QVHighlights<br>R1@.3','QVHighlights<br>R1@.5','QVHighlights<br>R1@.7','QVHighlights<br>mIoU','Average ↑','Latency (s) ↓']
      : ['Charades','ActivityNet','QVHighlights','Average ↑','Latency (s) ↓']
      : ['LVBench','Video-MME','LongVideoBench','MLVU','Average ↑','Latency (s) ↓'];
    const indices = ground ? metrics ? [...Array.from({length:13},(_,i)=>i+1),0] : [4,8,12,13,0] : [1,2,3,4,5,0];
    let previousGroup = '';
    const groupName = {'4B':'Dense · 4B','MoE':'MoE · 3B active','Proprietary':'Proprietary'};
    const body = rows.map(row => {
      let prefix = '';
      if (previousGroup !== row.size) {
        previousGroup = row.size;
        prefix = `<tr class="group-label"><th colspan="${headings.length+1}" scope="colgroup">${groupName[row.size]}</th></tr>`;
      }
      const ours = row.model.startsWith('VideoJev');
      const mode = row.size==='Proprietary'?'':`<span class="model-mode">${escape(row.mode)}</span>`;
      return `${prefix}<tr class="${ours?'ours':''}"><th scope="row">${escape(row.model)}${mode}</th>${indices.map((index,i) => `<td class="${i===indices.length-2?'average-cell':index===0?'latency-cell':''}">${row.values[index]===null?'—':row.values[index].toFixed(index===0?3:1)}</td>`).join('')}</tr>`;
    }).join('');
    $('#benchmark-table').innerHTML = `<caption>${ground?'Temporal grounding scores.':'Multiple-choice video QA scores.'}</caption><thead><tr><th scope="col">Model / inference mode</th>${headings.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody>`;
    $('#table-note').textContent = ground
      ? (metrics ? 'mIoU and recall (%).' : 'Mean temporal IoU (mIoU, %).')
      : 'Accuracy (%); MLVU uses M-Avg. Higher is better.';
  }
  $$('[data-table-task]').forEach(button => button.addEventListener('click',() => {
    tableTask = button.dataset.tableTask;
    $$('[data-table-task]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    renderTable();
  }));
  ['#table-scale','#show-all-methods','#show-all-metrics'].forEach(selector=>$(selector).addEventListener('change',renderTable));
  renderTable();

  const dialog = $('#figure-dialog');
  let dialogTrigger = null;
  $$('[data-zoom]').forEach(button => button.addEventListener('click',() => {
    dialogTrigger = button;
    $('#dialog-image').src = button.dataset.zoom;
    $('#dialog-image').alt = button.dataset.caption || 'Expanded figure';
    $('#dialog-caption').textContent = button.dataset.caption;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  }));
  $('#close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
  dialog.addEventListener('close',()=>{document.body.classList.remove('dialog-open');dialogTrigger?.focus({preventScroll:true});});

  let copyReset;
  $('#copy-citation').addEventListener('click',async () => {
    const text = $('#bibtex').textContent;
    const button = $('#copy-citation');
    let copied = false;
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      const area = document.createElement('textarea');
      area.value = text;area.style.position='fixed';area.style.top='-9999px';
      document.body.append(area);area.select();
      copied = document.execCommand('copy');area.remove();
    }
    button.innerHTML = copied ? 'Copied <span aria-hidden="true">✓</span>' : 'Select citation below';
    $('#copy-status').textContent = copied ? 'BibTeX copied to clipboard.' : 'Clipboard unavailable. Select and copy the citation below.';
    if (!copied) {
      const range = document.createRange();range.selectNodeContents($('#bibtex'));
      const selection = window.getSelection();selection.removeAllRanges();selection.addRange(range);
    }
    clearTimeout(copyReset);
    copyReset = setTimeout(()=>{button.innerHTML='Copy BibTeX <span aria-hidden="true">⧉</span>';},2500);
  });


  const links = $$('.site-header nav a');
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach(link => {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle('active',active);
        if (active) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current');
      });
    }
  }, {rootMargin:'-15% 0px -65% 0px',threshold:0});
  ['overview','examples','method','results','models'].forEach(id=>observer.observe(document.getElementById(id)));
})();
