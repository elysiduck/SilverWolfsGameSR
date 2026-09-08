# -*- coding: utf-8 -*-
"""把 105 个 NPU benchmark 延迟数据渲染成单页 HTML（热力矩阵 + 全量表格）"""
import csv
import re
import html

CSV_PATH = r'D:\下载\本地传输\NPU-Sweep-105-Latency.csv'
OUT_HTML = r'D:\下载\本地传输\NPU-Sweep-105-Latency.html'

rows = []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        rows.append(r)

for r in rows:
    m = re.match(r'C(\d+)-L(\d+)', r['family'])
    r['c'] = int(m.group(1))
    r['l'] = int(m.group(2))
    r['avg'] = float(r['avg_ms'])

chans = sorted({r['c'] for r in rows})
layers = sorted({r['l'] for r in rows})
lut = {(r['c'], r['l']): r for r in rows}

avgs = [r['avg'] for r in rows]
vmin, vmax = min(avgs), max(avgs)


def heat(v):
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
    # 深青 -> 黄 -> 红（慢=红）
    if t < 0.5:
        k = t / 0.5
        r_ = int(20 + k * (250 - 20))
        g = int(90 + k * (205 - 90))
        b = int(80 + k * (60 - 80))
    else:
        k = (t - 0.5) / 0.5
        r_ = int(250 + k * (232 - 250))
        g = int(205 + k * (72 - 205))
        b = int(60 + k * (72 - 60))
    return '#%02x%02x%02x' % (r_, g, b)


matrix_rows = []
for c in chans:
    tds = ''.join(
        '<td class="cell" style="background:%s" data-v="%.4f">%.3f</td>' % (
            heat(lut[(c, l)]['avg']) if (c, l) in lut else '#222', lut[(c, l)]['avg'],
            lut[(c, l)]['avg']) for l in layers)
    matrix_rows.append('<tr><th>C%d</th>%s</tr>' % (c, tds))

table_rows = []
order = sorted(rows, key=lambda r: r['avg'])
for i, r in enumerate(order, 1):
    table_rows.append(
        '<tr data-avg="{avg:.4f}" data-c="{c}" data-l="{l}">'
        '<td class="dim">{i}</td><td class="fam">{fam}</td>'
        '<td class="num strong" style="color:{col}">{avg:.4f}</td>'
        '<td class="num">{fps:.2f}</td>'
        '<td class="num">{p50:.4f}</td><td class="num">{p95:.4f}</td><td class="num">{p99:.4f}</td>'
        '<td class="num">{mn:.4f}</td><td class="num">{mx:.4f}</td>'
        '<td class="num">{first}</td><td class="num">{sess}</td>'
        '<td class="dim">{st}</td><td class="dim">{npu}%</td></tr>'.format(
            avg=r['avg'], c=r['c'], l=r['l'], i=i, fam=html.escape(r['family']),
            col=heat(r['avg']), fps=float(r['fps']),
            p50=float(r['p50_ms']), p95=float(r['p95_ms']), p99=float(r['p99_ms']),
            mn=float(r['min_ms']), mx=float(r['max_ms']),
            first=r['first_ms'], sess=r['session_s'],
            st=r['status'], npu=r['npu_events']))

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GameSR NPU 架构扫描 - 105 模型延迟</title>
<style>
:root{--bg:#14161a;--panel:#1c1f26;--line:#2b303b;--txt:#e6e9ef;--dim:#8b93a3;--acc:#4dd0e1}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--txt);
 font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
 padding:18px;margin-bottom:20px;overflow-x:auto}
h2{font-size:15px;margin:0 0 14px;color:var(--acc);font-weight:600}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--panel)}
th:first-child,td:first-child{text-align:left}
.matrix th{background:var(--panel);color:var(--txt);font-weight:600}
.matrix td.cell{text-align:center;font-size:12px;color:#0d0f12;font-weight:600;border-radius:3px}
.fam{font-family:Consolas,monospace;font-weight:600}
.num{font-family:Consolas,monospace}
.strong{font-weight:700}
.dim{color:var(--dim);font-size:12px}
.wrap{max-height:70vh;overflow-y:auto}
.kv{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:14px;font-size:12.5px;color:var(--dim)}
.kv b{color:var(--txt);font-family:Consolas,monospace}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim);margin-top:12px}
.bar{height:10px;width:220px;border-radius:5px;
 background:linear-gradient(90deg,#145a50,#f0a83c,#e84848)}
</style></head><body>
<h1>GameSR NPU 架构扫描 · 105 个模型延迟</h1>
<div class="sub">数据源 GameSR-NPU-Architecture-Sweep-20260905-161928-COMPACT.zip（SweepRunner 日志 105 份）
 · 设备 Ryzen AI NPU (X1-Benchmark) · int8 QDQ / VitisAI EP · warmup 20 · 每模型 600 次串行推理 ·
 仅模型推理耗时，不含采集/预处理/显示/Magpie</div>

<div class="kv">
<span>模型数 <b>__N__</b></span>
<span>最快 <b>__MIN__ ms</b></span>
<span>最慢 <b>__MAX__ ms</b></span>
<span>中位 <b>__MED__ ms</b></span>
<span>全部 <b>NPU_ONLY</b>（NPU 事件/时长占比 100%）</span>
</div>

<div class="card">
<h2>延迟热力矩阵（平均推理耗时 ms，绿=快 红=慢）</h2>
<table class="matrix"><thead><tr><th>C\\L</th>__LH__</tr></thead>
<tbody>__MX__</tbody></table>
<div class="legend"><span>__VMIN__ ms</span><div class="bar"></div><span>__VMAX__ ms</span></div>
</div>

<div class="card">
<h2>全量 105 条（默认按平均延迟升序，点击表头可排序）</h2>
<div class="wrap"><table id="t">
<thead><tr>
<th data-k="idx">#</th><th data-k="fam">模型</th>
<th data-k="avg">平均 ms</th><th data-k="fps">FPS</th>
<th data-k="p50">P50 ms</th><th data-k="p95">P95 ms</th><th data-k="p99">P99 ms</th>
<th data-k="min">Min ms</th><th data-k="max">Max ms</th>
<th data-k="first">首帧 ms</th><th data-k="sess">建会话 s</th>
<th data-k="st">状态</th><th data-k="npu">NPU%</th>
</tr></thead>
<tbody>__TB__</tbody></table></div>
</div>

<script>
const tb=document.querySelector('#t tbody');
const K={idx:r=>+r.dataset.i,fam:r=>r.children[1].textContent,
 avg:r=>+r.dataset.avg,fps:r=>+r.children[3].textContent,
 p50:r=>+r.children[4].textContent,p95:r=>+r.children[5].textContent,
 p99:r=>+r.children[6].textContent,min:r=>+r.children[7].textContent,
 max:r=>+r.children[8].textContent,first:r=>+r.children[9].textContent,
 sess:r=>+r.children[10].textContent,st:r=>r.children[11].textContent,
 npu:r=>+r.children[12].textContent};
let dir={};
[...document.querySelectorAll('#t th')].forEach((th,i)=>{th.style.cursor='pointer';
 th.onclick=()=>{const k=th.dataset.k;dir[k]=!dir[k];
  const rs=[...tb.rows];rs.sort((a,b)=>{let x=K[k](a),y=K[k](b);
   if(typeof x==='string')return dir[k]?x.localeCompare(y):y.localeCompare(x);
   return dir[k]?x-y:y-x;});rs.forEach(r=>tb.appendChild(r));};});
</script>
</body></html>"""

HTML = (HTML
        .replace('__N__', str(len(rows)))
        .replace('__MIN__', '%.4f' % vmin)
        .replace('__MAX__', '%.4f' % vmax)
        .replace('__MED__', '%.4f' % sorted(avgs)[len(avgs) // 2])
        .replace('__VMIN__', '%.2f' % vmin)
        .replace('__VMAX__', '%.2f' % vmax)
        .replace('__LH__', ''.join('<th>L%d</th>' % l for l in layers))
        .replace('__MX__', '\n'.join(matrix_rows))
        .replace('__TB__', '\n'.join(table_rows)))

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(HTML)
print('OK ->', OUT_HTML, len(rows), 'rows')
