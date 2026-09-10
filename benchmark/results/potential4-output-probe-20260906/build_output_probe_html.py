# -*- coding: utf-8 -*-
"""把两组 NPU output 延迟探针渲染成交互式 HTML 看板（热力矩阵 + 可排序表，含 avg/P50/P95/增量）。
数据源（从 GitHub 下载到工作区）：
  GameSR-Output-Channel-Probe-20260906.csv   (C16-L4 固定，扫输出通道 O3/O4/O6/O8)
  GameSR-Potential4-Output-Probe-20260906.csv (4 主干 × O3/O4/O6/O8 = 16 模型)
产出：
  NPU-Output-Channel-Probe-Dashboard.html
  NPU-Potential4-Output-Probe-Dashboard.html
"""
import csv
import html

SRC_OCP = r'D:\下载\本地传输\GameSR-Output-Channel-Probe-20260906.csv'
SRC_P4 = r'D:\下载\本地传输\GameSR-Potential4-Output-Probe-20260906.csv'
OUT_OCP = r'D:\下载\本地传输\NPU-Output-Channel-Probe-Dashboard.html'
OUT_P4 = r'D:\下载\本地传输\NPU-Potential4-Output-Probe-Dashboard.html'

# ---------- 调色（与 105 看板一致）----------
def heat_avg(v, vmin, vmax):
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
    if t < 0.5:
        k = t / 0.5
        r = int(20 + k * (250 - 20)); g = int(90 + k * (205 - 90)); b = int(80 + k * (60 - 80))
    else:
        k = (t - 0.5) / 0.5
        r = int(250 + k * (232 - 250)); g = int(205 + k * (72 - 205)); b = int(60 + k * (72 - 60))
    return '#%02x%02x%02x' % (r, g, b)

def heat_delta(d):
    # d: 相对 O3 的增量百分比（0% 绿 -> +15% 红）
    t = max(0.0, min(1.0, d / 15.0))
    r = int(40 + t * (220 - 40)); g = int(180 + t * (70 - 180)); b = int(90 + t * (60 - 90))
    return '#%02x%02x%02x' % (r, g, b)

CSS = """
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
.matrix td.cell{text-align:center;color:#0d0f12;border-radius:4px;vertical-align:middle}
.cell .big{font-size:14px;font-weight:700}
.cell .small{font-size:10px;opacity:.8;font-weight:600}
.fam{font-family:Consolas,monospace;font-weight:600}
.num{font-family:Consolas,monospace}
.strong{font-weight:700}
.dim{color:var(--dim);font-size:12px}
.wrap{max-height:70vh;overflow-y:auto}
.kv{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:14px;font-size:12.5px;color:var(--dim)}
.kv b{color:var(--txt);font-family:Consolas,monospace}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim);margin-top:12px}
.bar{height:10px;width:220px;border-radius:5px}
.note{background:#1a1f17;border:1px solid #2c3a26;border-radius:8px;padding:12px 14px;
 margin-bottom:20px;color:#bcd0a8;font-size:12.5px;line-height:1.6}
.note b{color:#e6f0d8}
"""

def cell_avg(avg, p95, vmin, vmax):
    return ('<td class="cell" style="background:%s">'
            '<div class="big">%.3f</div><div class="small">P95 %.3f</div></td>') % (
        heat_avg(avg, vmin, vmax), avg, p95)

def sortable_script(ncols):
    return """
<script>
const tb=document.querySelector('#t tbody');
const K={};
[...document.querySelectorAll('#t th')].forEach((th,i)=>{th.style.cursor='pointer';
 th.onclick=()=>{const k=th.dataset.k;const d=!(th.dataset.asc==='1');
  th.dataset.asc=d?'1':'0';
  const rs=[...tb.rows];rs.sort((a,b)=>{let x=a.dataset[k],y=b.dataset[k];
   if(x!==undefined&&!isNaN(+x)){x=+x;y=+y;return d?x-y:y-x;}
   return d?x.localeCompare(y):y.localeCompare(x);});
  rs.forEach(r=>tb.appendChild(r));};});
</script>"""

# =================== 1) output-channel-probe ===================
ocp = []
with open(SRC_OCP, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        ocp.append({
            'o': int(r['output_channels']),
            'elems': int(float(r['output_elements'])),
            'mib': float(r['output_fp32_mib']),
            'avg': float(r['average_ms']),
            'p50': float(r['p50_ms']),
            'p95': float(r['p95_ms']),
            'd': float(r['delta_percent_vs_o3']),
            'fps': float(r['pure_fps']),
            'gmac': float(r['effective_gmac_s']),
            'st': r['status'],
        })
ocp.sort(key=lambda x: x['o'])
vmin = min(x['avg'] for x in ocp); vmax = max(x['avg'] for x in ocp)

# 1x4 热力行（avg + p95）
heatrow = ''.join(cell_avg(x['avg'], x['p95'], vmin, vmax) for x in ocp)
matrix_ocp_head = '<th>输出宽度</th>' + ''.join('<th>O%d</th>' % x['o'] for x in ocp)
matrix_ocp_body = '<tr><th>C16-L4</th>%s</tr>' % heatrow

rows_ocp = []
for i, x in enumerate(ocp, 1):
    rows_ocp.append(
        '<tr data-o="{o}" data-avg="{avg}" data-p50="{p50}" data-p95="{p95}" data-d="{d}">'
        '<td class="dim">{i}</td><td class="fam">O{o}</td>'
        '<td class="num">{elems:,}</td><td class="num">{mib:.2f}</td>'
        '<td class="num strong" style="color:{col}">{avg:.4f}</td>'
        '<td class="num">{p50:.4f}</td><td class="num">{p95:.4f}</td>'
        '<td class="num">{d:+.2f}%</td><td class="num">{fps:.2f}</td>'
        '<td class="num">{gmac:.1f}</td><td class="dim">{st}</td></tr>'.format(
            o=x['o'], avg=x['avg'], p50=x['p50'], p95=x['p95'], d=x['d'], i=i,
            elems=x['elems'], mib=x['mib'], col=heat_avg(x['avg'], vmin, vmax),
            fps=x['fps'], gmac=x['gmac'], st=x['st']))

HTML_OCP = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GameSR NPU 输出通道探针 · C16-L4</title>
<style>__CSS__</style></head><body>
<h1>GameSR NPU 输出通道探针 · C16-L4（固定主干）</h1>
<div class="sub">数据源 GameSR-NPU-Output-Channel-Probe-20260906-121534-COMPACT.zip
 · 仅改最后一层输出宽度（输出通道 O3→O8）以隔离输出成本
 · 输入 1×3×720×1280 · warmup 5 / runs 120 · 设备 Ryzen AI NPU (X1-Benchmark) · int8 QDQ / VitisAI EP</div>

<div class="note"><b>范围说明：</b>本探针测的是<b>纯 NPU 运行时</b>，不含 Magpie 的 Map/stitch/upload，也不含 learned GPU decoder。
真实端到端"输出延迟"会更高（输出张量回传主机/上采样那段未计入）。所有模型 <b>status=NPU_ONLY</b>，NPU 事件/时长占比 100%。</div>

<div class="kv">
<span>配置数 <b>__N__</b></span>
<span>输出通道 <b>O3 / O4 / O6 / O8</b></span>
<span>最快 <b>__MIN__ ms</b> (O__MINO__)</span>
<span>最慢 <b>__MAX__ ms</b> (O__MAXO__)</span>
<span>全部 <b>NPU_ONLY</b></span>
</div>

<div class="card">
<h2>平均推理耗时热力条（ms，绿=快 红=慢；每格下方为 P95）</h2>
<table class="matrix"><thead><tr>__MH__</tr></thead><tbody>__MR__</tbody></table>
<div class="legend"><span>__VMIN__ ms</span><div class="bar" style="background:linear-gradient(90deg,#145a50,#f0a83c,#e84848)"></div><span>__VMAX__ ms</span></div>
</div>

<div class="card">
<h2>全量 __N__ 条（点击表头可排序）</h2>
<div class="wrap"><table id="t">
<thead><tr>
<th data-k="o">#</th><th data-k="o">输出通道</th>
<th data-k="elems">输出元素</th><th data-k="mib">输出MiB</th>
<th data-k="avg">平均 ms</th><th data-k="p50">P50 ms</th><th data-k="p95">P95 ms</th>
<th data-k="d">vs O3</th><th data-k="fps">FPS</th><th data-k="gmac">GMAC/s</th>
<th data-k="st">状态</th>
</tr></thead>
<tbody>__TB__</tbody></table></div>
</div>
__SCRIPT__
</body></html>"""

HTML_OCP = (HTML_OCP.replace('__CSS__', CSS)
    .replace('__N__', str(len(ocp)))
    .replace('__MIN__', '%.4f' % vmin).replace('__MINO__', str(min(ocp,key=lambda x:x['avg'])['o']))
    .replace('__MAX__', '%.4f' % vmax).replace('__MAXO__', str(max(ocp,key=lambda x:x['avg'])['o']))
    .replace('__VMIN__', '%.2f' % vmin).replace('__VMAX__', '%.2f' % vmax)
    .replace('__MH__', matrix_ocp_head)
    .replace('__MR__', matrix_ocp_body)
    .replace('__TB__', '\n'.join(rows_ocp))
    .replace('__SCRIPT__', sortable_script(11)))

with open(OUT_OCP, 'w', encoding='utf-8') as f:
    f.write(HTML_OCP)
print('OK ->', OUT_OCP, len(ocp), 'rows')

# =================== 2) potential4-output-probe ===================
p4 = []
with open(SRC_P4, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        p4.append({
            'bb': r['backbone'],
            'c': int(r['channels']),
            'l': int(r['conv_layers']),
            'o': int(r['output_channels']),
            'arch': r['architecture'],
            'avg': float(r['average_ms']),
            'p50': float(r['p50_ms']),
            'p95': float(r['p95_ms']),
            'd': float(r['delta_percent_vs_same_backbone_o3']),
            'mib': float(r['output_fp32_mib']),
            'st': r['status'],
        })

backbones = ['C8-L5', 'C16-L4', 'C24-L3', 'C32-L3']
os_ = [3, 4, 6, 8]
lut = {(x['bb'], x['o']): x for x in p4}
vmin = min(x['avg'] for x in p4); vmax = max(x['avg'] for x in p4)
dmax = max(x['d'] for x in p4)

# avg 矩阵
m_avg = []
for bb in backbones:
    tds = ''.join(cell_avg(lut[(bb, o)]['avg'], lut[(bb, o)]['p95'], vmin, vmax) for o in os_)
    m_avg.append('<tr><th>%s</th>%s</tr>' % (bb, tds))
# delta 矩阵
m_delta = []
for bb in backbones:
    tds = ''.join('<td class="cell" style="background:%s"><div class="big">%+.2f%%</div></td>' % (
        heat_delta(lut[(bb, o)]['d']), lut[(bb, o)]['d']) for o in os_)
    m_delta.append('<tr><th>%s</th>%s</tr>' % (bb, tds))

rows_p4 = []
for i, x in enumerate(sorted(p4, key=lambda r: (backbones.index(r['bb']), r['o'])), 1):
    rows_p4.append(
        '<tr data-bb="{bb}" data-c="{c}" data-l="{l}" data-o="{o}" data-avg="{avg}" '
        'data-p50="{p50}" data-p95="{p95}" data-d="{d}">'
        '<td class="dim">{i}</td><td class="fam">{bb}</td>'
        '<td class="num">{c}</td><td class="num">{l}</td><td class="fam">O{o}</td>'
        '<td class="num">{arch}</td>'
        '<td class="num strong" style="color:{col}">{avg:.4f}</td>'
        '<td class="num">{p50:.4f}</td><td class="num">{p95:.4f}</td>'
        '<td class="num">{d:+.2f}%</td><td class="num">{mib:.2f}</td><td class="dim">{st}</td></tr>'.format(
            bb=x['bb'], c=x['c'], l=x['l'], o=x['o'], arch=html.escape(x['arch']), i=i,
            avg=x['avg'], p50=x['p50'], p95=x['p95'], d=x['d'], mib=x['mib'],
            col=heat_avg(x['avg'], vmin, vmax), st=x['st']))

HTML_P4 = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GameSR NPU Potential-4 输出探针</title>
<style>__CSS__</style></head><body>
<h1>GameSR NPU Potential-4 输出探针</h1>
<div class="sub">数据源 GameSR-NPU-Potential4-Output-Probe-20260906-124724-COMPACT.zip
 · 4 主干 × 输出通道 O3/O4/O6/O8 = 16 模型 · 输入 1×3×720×1280 · warmup 20 / runs 600
 · 设备 Ryzen AI NPU (X1-Benchmark) · int8 QDQ / VitisAI EP</div>

<div class="note"><b>范围说明：</b>纯 NPU 运行时，不含 Magpie Map/stitch/upload 与 learned GPU decoder。
增量为"相对<b>同主干 O3</b>"。所有模型 <b>status=NPU_ONLY</b>。这是<b>随机权重结构探针</b>（非训练模型），
绝对延迟不可直接与 105 模型 architecture-sweep 比，仅用于看"输出宽度带来的相对增量"。</div>

<div class="kv">
<span>模型数 <b>__N__</b></span>
<span>主干 <b>C8-L5 / C16-L4 / C24-L3 / C32-L3</b></span>
<span>输出通道 <b>O3 / O4 / O6 / O8</b></span>
<span>最快 <b>__MIN__ ms</b></span>
<span>最慢 <b>__MAX__ ms</b></span>
<span>全部 <b>NPU_ONLY</b></span>
</div>

<div class="card">
<h2>平均推理耗时热力矩阵（ms，绿=快 红=慢；每格下方为 P95）</h2>
<table class="matrix"><thead><tr><th>主干 \\ O</th>__OH__</tr></thead><tbody>__MAVG__</tbody></table>
<div class="legend"><span>__VMIN__ ms</span><div class="bar" style="background:linear-gradient(90deg,#145a50,#f0a83c,#e84848)"></div><span>__VMAX__ ms</span></div>
</div>

<div class="card">
<h2>相对同主干 O3 的延迟增量（%，绿=0 红=+15%）</h2>
<table class="matrix"><thead><tr><th>主干 \\ O</th>__OH__</tr></thead><tbody>__MDELTA__</tbody></table>
<div class="legend"><span>0%</span><div class="bar" style="background:linear-gradient(90deg,#28b45a,#f0a83c,#e84848)"></div><span>+__DMAX__%</span></div>
</div>

<div class="card">
<h2>全量 16 条（点击表头可排序）</h2>
<div class="wrap"><table id="t">
<thead><tr>
<th data-k="i">#</th><th data-k="bb">主干</th>
<th data-k="c">通道C</th><th data-k="l">层数L</th><th data-k="o">输出</th>
<th data-k="arch">架构</th>
<th data-k="avg">平均 ms</th><th data-k="p50">P50 ms</th><th data-k="p95">P95 ms</th>
<th data-k="d">vs O3</th><th data-k="mib">输出MiB</th><th data-k="st">状态</th>
</tr></thead>
<tbody>__TB__</tbody></table></div>
</div>
__SCRIPT__
</body></html>"""

HTML_P4 = (HTML_P4.replace('__CSS__', CSS)
    .replace('__N__', str(len(p4)))
    .replace('__MIN__', '%.4f' % vmin).replace('__MAX__', '%.4f' % vmax)
    .replace('__VMIN__', '%.2f' % vmin).replace('__VMAX__', '%.2f' % vmax)
    .replace('__OH__', ''.join('<th>O%d</th>' % o for o in os_))
    .replace('__MAVG__', '\n'.join(m_avg))
    .replace('__DMAX__', '%.1f' % dmax)
    .replace('__MDELTA__', '\n'.join(m_delta))
    .replace('__TB__', '\n'.join(rows_p4))
    .replace('__SCRIPT__', sortable_script(12)))

with open(OUT_P4, 'w', encoding='utf-8') as f:
    f.write(HTML_P4)
print('OK ->', OUT_P4, len(p4), 'rows')
