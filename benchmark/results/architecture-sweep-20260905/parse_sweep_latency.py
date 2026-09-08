# -*- coding: utf-8 -*-
"""解析 GameSR NPU 架构扫描 105 模型的延迟数据 -> CSV"""
import zipfile
import re
import csv
import os

ZIP = r'D:\下载\本地传输\GameSR-NPU-Architecture-Sweep-20260905-161928-COMPACT.zip'
OUT = r'D:\下载\本地传输\NPU-Sweep-105-Latency.csv'

PATTERNS = [
    ('family',     re.compile(r'^Family\s*:\s*(\S+)', re.M)),
    ('avg_ms',     re.compile(r'Average inference time cost:\s*([\d.]+)\s*ms', re.I)),
    ('fps',        re.compile(r'Number of inferences per second:\s*([\d.]+)', re.I)),
    ('first_ms',   re.compile(r'First inference time cost:\s*(\d+)\s*ms', re.I)),
    ('session_s',  re.compile(r'Session creation time cost:\s*([\d.]+)\s*s', re.I)),
    ('min_ms',     re.compile(r'^Min Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('max_ms',     re.compile(r'^Max Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('p50_ms',     re.compile(r'^P50 Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('p90_ms',     re.compile(r'^P90 Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('p95_ms',     re.compile(r'^P95 Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('p99_ms',     re.compile(r'^P99 Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('p999_ms',    re.compile(r'^P999 Latency:\s*([\d.eE+-]+)\s*s', re.M | re.I)),
    ('status',     re.compile(r'int8/vitisai: status=([A-Z_]+)')),
    ('npu_events', re.compile(r'NPU-events=(\d+)%')),
    ('npu_dur',    re.compile(r'NPU-duration=(\d+)%')),
    ('cpu_pct',    re.compile(r'Avg CPU usage:\s*(\d+)\s*%')),
    ('ws_mb',      re.compile(r'Peak working set size:\s*(\d+)\s*bytes')),
    ('runs',       re.compile(r'Runs:\s*(\d+)')),
    ('sha',        re.compile(r'SHA-256 fingerprint:\s*([0-9a-f]{16})')),
]

SEC_KEYS = ('min_ms', 'max_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'p999_ms')


def main():
    z = zipfile.ZipFile(ZIP)
    names = sorted(n for n in z.namelist() if n.startswith('SweepRunner'))
    rows = []
    for n in names:
        raw = z.read(n).decode('utf-8-sig', 'replace')
        r = {'log': n}
        for key, pat in PATTERNS:
            m = pat.search(raw)
            r[key] = m.group(1) if m else ''
        for k in SEC_KEYS:
            if r[k]:
                r[k] = round(float(r[k]) * 1000.0, 4)
        for k in ('avg_ms', 'fps', 'session_s'):
            if r[k]:
                r[k] = round(float(r[k]), 4)
        if r['first_ms']:
            r['first_ms'] = int(r['first_ms'])
        if r['ws_mb']:
            r['ws_mb'] = round(int(r['ws_mb']) / 1048576.0, 1)
        rows.append(r)

    def sort_key(r):
        fam = r['family'] or ''
        m = re.match(r'C(\d+)-L(\d+)', fam)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    rows.sort(key=sort_key)

    cols = ['family', 'avg_ms', 'fps', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms',
            'p999_ms', 'min_ms', 'max_ms', 'first_ms', 'session_s',
            'status', 'npu_events', 'npu_dur', 'cpu_pct', 'ws_mb', 'runs', 'sha']
    with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    missing = [r['family'] for r in rows if not r['avg_ms']]
    print('parsed=%d  missing_avg=%d %s' % (len(rows), len(missing), missing[:10]))
    print('written ->', OUT)

    # 控制台输出：按 C x L 矩阵
    chans = sorted({int(re.match(r'C(\d+)', r['family']).group(1)) for r in rows if r['family']})
    layers = sorted({int(re.search(r'-L(\d+)', r['family']).group(1)) for r in rows if r['family']})
    lut = {(int(re.match(r'C(\d+)', r['family']).group(1)),
            int(re.search(r'-L(\d+)', r['family']).group(1))): r['avg_ms']
           for r in rows if r['family']}
    print('\n平均延迟 avg (ms)  C=通道 L=层数')
    print('     ' + ''.join('%9s' % ('L%d' % l) for l in layers))
    for c in chans:
        print('C%-4d' % c + ''.join('%9s' % (lut.get((c, l), '-') if lut.get((c, l)) else '-') for l in layers))

    ok = [r for r in rows if r['avg_ms']]
    ok.sort(key=lambda r: float(r['avg_ms']))
    print('\n最快 10:')
    for r in ok[:10]:
        print('  %-8s %8s ms  %8s FPS' % (r['family'], r['avg_ms'], r['fps']))
    print('最慢 5:')
    for r in ok[-5:]:
        print('  %-8s %8s ms  %8s FPS' % (r['family'], r['avg_ms'], r['fps']))


if __name__ == '__main__':
    main()
