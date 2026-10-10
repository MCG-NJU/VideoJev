"""Prepare the static project page from the paper's versioned source assets."""
from pathlib import Path
import json, re, shutil, subprocess
import pymupdf
from PIL import Image
from fontTools.ttLib import TTFont

ROOT = Path('/pfs/zhuyuhan/workspace/paper_work/VideoJev')
OUT = Path(__file__).parent
ASSETS = OUT / 'assets'
for name in ['figures', 'frames', 'videos', 'fonts', 'data']:
    (ASSETS/name).mkdir(parents=True, exist_ok=True)

fonts = ROOT/'figures/video-jev-charades-v48/fonts'
shutil.copy2(fonts/'Inter-Regular.woff2', ASSETS/'fonts/inter-regular.woff2')
shutil.copy2(fonts/'OFL.txt', ASSETS/'fonts/OFL.txt')
for source, target in [('Inter-Medium.ttf', 'inter-medium'), ('Inter-Bold.ttf', 'inter-bold')]:
    font = TTFont(fonts/source)
    font.flavor = 'woff2'
    font.save(ASSETS/f'fonts/{target}.woff2')
shutil.copy2(ROOT/'main.pdf', ASSETS/'VideoJev.pdf')

def render(source, name, width=2300, region=None):
    page = pymupdf.open(ROOT/source)[0]
    rect = page.rect
    if region:
        rect = pymupdf.Rect(rect.width*region[0], rect.height*region[1], rect.width*region[2], rect.height*region[3])
    pix = page.get_pixmap(matrix=pymupdf.Matrix(width/rect.width, width/rect.width), clip=rect, alpha=False)
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    image.save(ASSETS/f'figures/{name}.webp', quality=91, method=6)

render('figures/video-jev-teaser/video-jev-teaser.pdf', 'teaser')
render('figures/video-jev-teaser/video-jev-teaser.pdf', 'performance-grounding', 1600, (0, .245, .505, 1))
render('figures/video-jev-teaser/video-jev-teaser.pdf', 'performance-qa', 1600, (.505, .245, 1, 1))
render('figures/video-jev-charades-v48/video-jev-charades-v48.pdf', 'architecture', 2800)
for i in range(3):
    render('figures/video-jev-charades-v48/video-jev-charades-v48.pdf', f'architecture-{i+1}', 1100, (i/3, 0, (i+1)/3, 1))
render('figures/query_scaling_4b.pdf', 'query-scaling', 2600)
render('figures/query_scaling_4b.pdf', 'scaling-grounding', 1700, (0, 0, .50, .885))
render('figures/query_scaling_4b.pdf', 'scaling-qa', 1700, (.50, 0, 1, .885))
render('figures/qa_calibration.pdf', 'calibration', 2400)
for a,b in [(1,2),(3,4),(5,6),(7,8)]:
    render(f'figures/qualitative_demos/cases_{a:02d}_{b:02d}.pdf', f'cases-{a}-{b}', 1900)

source_cases = json.loads((ROOT/'research/qualitative_demos/selected_cases.json').read_text())['cases']
cases = []
for case in source_cases:
    n = case['number']
    frames = []
    for i, time in enumerate(case['contact_times']):
        name = f'case-{n:02d}-{i:02d}.webp'
        im = Image.open(ROOT/f'research/qualitative_demos/frames/case_{n:02d}/frame_{i:03d}.jpg')
        im.thumbnail((1280, 800))
        im.save(ASSETS/'frames'/name, quality=86, method=6)
        frames.append({'time':time, 'src':f'assets/frames/{name}'})
    def clean_query(q):
        a = q['annotation']
        return {'query': a['query'], 'options': a.get('options'), 'answer': a.get('answer_index'),
                'spans':a.get('spans'), 'predictions': {size:{k:v for k,v in pred.items() if k in ['prediction','probabilities','answer_index','correct']} for size,pred in q['predictions'].items()}}
    cases.append({'id':n, 'title':case['capability'], 'duration':case['annotation']['duration'],
                  'dataset':case['annotation'].get('source',case['annotation'].get('benchmark', '')),
                  'frames':frames, 'queries':[clean_query(q) for q in case.get('multi_queries',[case])],
                  'interpretation':case['interpretation'], 'groupSize': next(iter(case.get('joint_group',{}).values()),{}).get('question_count')})
    if n in [1,4]:
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2']
        cmd += ['-i',case['annotation']['video_path']]
        if n==4: cmd += ['-t','48']
        cmd += ['-an','-vf','scale=960:-2','-c:v','libx264','-preset','fast','-crf','24','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart', str(ASSETS/f'videos/case-{n:02d}.mp4')]
        subprocess.run(cmd, check=True)

def clean(s):
    s = re.sub(r'\\textsubscript\{(Instruct|Thinking)\}',r' \1',s)
    s = re.sub(r'~\\citep\{[^}]*\}','',s)
    s = re.sub(r'\\(?:color|rowcolor)\{[^}]*\}','',s)
    s = re.sub(r'\\(?:bfseries|leavevmode|tblSecond|uline|textbf)','',s)
    return s.replace('{','').replace('}','').replace('\\\\','').strip()

tex = (ROOT/'sections/04_experiments.tex').read_text()
tables = {}
for task,label,count in [('grounding','tab:temporal-grounding',15),('qa','tab:video-qa',7)]:
    block = tex.split('\\label{'+label+'}')[1].split('\\begin{tabular}',1)[1].split('\\end{tabular}',1)[0]
    rows=[]
    for line in block.splitlines():
        if line.lstrip().startswith('%') or '&' not in line:continue
        cells=[clean(x) for x in line.split('&')]
        if len(cells)!=count or not cells[0] or cells[0].startswith('\\'):continue
        try: values=[None if c=='--' else float(c) for c in cells[1:]]
        except ValueError:continue
        raw = cells[0]
        name = re.sub(r' (Thinking|Instruct)$','',raw)
        mode = 'Joint' if name.startswith('VideoJev') else 'Thinking' if raw.endswith('Thinking') else 'Instruct'
        size = '4B' if (name.endswith('4B') or name=='Mage-VL') else 'MoE' if 'A3B' in name else 'Proprietary'
        rows.append({'model':name, 'mode':mode, 'size':size, 'values':values})
    tables[task]=rows
assert len(tables['grounding'])==15, len(tables['grounding'])
assert len(tables['qa'])==20, len(tables['qa'])
data={'cases':cases,'tables':tables}
(ASSETS/'data/paper-data.json').write_text(json.dumps(data,indent=2))
(ASSETS/'data/paper-data.js').write_text('window.PAPER_DATA = '+json.dumps(data)+';\n')
print('Prepared',len(cases),'cases;', {k:len(v) for k,v in tables.items()}, 'table rows.')
print('Asset size:',sum(p.stat().st_size for p in ASSETS.rglob('*') if p.is_file())//1024,'KiB')
