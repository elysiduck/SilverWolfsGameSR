from __future__ import annotations
import csv, ctypes, json, re, time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
import dxcam, psutil
from PIL import Image

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
CAP, DATA, DETECT = CFG['capture'], CFG['dataset'], CFG['detection']
RAW_ROOT = ROOT / DATA['raw_root']
META = ROOT / DATA['metadata_csv']
RAW_ROOT.mkdir(parents=True, exist_ok=True)
META.parent.mkdir(parents=True, exist_ok=True)

U = ctypes.WinDLL('user32', use_last_error=True)
U.GetForegroundWindow.restype = wintypes.HWND
U.GetWindowTextLengthW.argtypes = [wintypes.HWND]
U.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
U.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
U.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
U.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
U.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
U.MonitorFromWindow.restype = wintypes.HMONITOR
U.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]

# Per-monitor DPI awareness prevents logical/physical pixel mismatches on scaled displays.
try:
    U.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    U.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
    U.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
except Exception:
    pass

class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('rcMonitor', wintypes.RECT),
        ('rcWork', wintypes.RECT),
        ('dwFlags', wintypes.DWORD),
    ]

BLACKLIST = {x.lower() for x in DETECT.get('ignore_processes', [])}


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s).strip()
    return s[:80] or 'UnknownGame'


def monitor_rect(hwnd: wintypes.HWND):
    mon = U.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
    if not mon:
        return None
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    if not U.GetMonitorInfoW(mon, ctypes.byref(mi)):
        return None
    r = mi.rcMonitor
    return int(r.left), int(r.top), int(r.right), int(r.bottom)


def fg_info():
    hwnd = U.GetForegroundWindow()
    if not hwnd:
        return None
    n = U.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(max(n + 1, 1))
    U.GetWindowTextW(hwnd, buf, len(buf))
    title = buf.value.strip()
    pid = wintypes.DWORD()
    U.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        proc = psutil.Process(pid.value)
        name = proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    if not title or name.lower() in BLACKLIST:
        return None

    rect = wintypes.RECT()
    p = wintypes.POINT(0, 0)
    if not U.GetClientRect(hwnd, ctypes.byref(rect)) or not U.ClientToScreen(hwnd, ctypes.byref(p)):
        return None
    w, h = int(rect.right - rect.left), int(rect.bottom - rect.top)
    if w < CAP['min_window_width'] or h < CAP['min_window_height']:
        return None

    mon = monitor_rect(hwnd)
    if mon is None:
        return None

    return {
        'hwnd': hwnd,
        'pid': int(pid.value),
        'process': name,
        'title': title,
        'left': int(p.x),
        'top': int(p.y),
        'right': int(p.x + w),
        'bottom': int(p.y + h),
        'width': w,
        'height': h,
        'monitor_left': mon[0],
        'monitor_top': mon[1],
        'monitor_right': mon[2],
        'monitor_bottom': mon[3],
    }


def write_row(row):
    new = not META.exists()
    with META.open('a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def crop_from_output(frame, info):
    # dxcam returns the configured output, not desktop-global coordinates.
    # Translate the game client rectangle to coordinates relative to that monitor.
    ox, oy = info['monitor_left'], info['monitor_top']
    x0 = int(info['left'] - ox)
    y0 = int(info['top'] - oy)
    x1 = int(info['right'] - ox)
    y1 = int(info['bottom'] - oy)

    fh, fw = frame.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(fw, x1), min(fh, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, (x0, y0, x1, y1), (fw, fh)
    cropped = frame[cy0:cy1, cx0:cx1]
    return cropped, (x0, y0, x1, y1), (fw, fh)


def make_camera():
    return dxcam.create(
        output_idx=int(CAP.get('output_idx', 0)),
        output_color='RGB',
        processor_backend='numpy'
    )


def main():
    print('GameSR Collector V3.2 | capture-only mode | cleaner remains offline')
    print('Press Ctrl+C to stop.')
    cam = make_camera()
    game = None
    pid = None
    session = None
    seq = 0
    last_output_shape = None

    while True:
        start = time.monotonic()
        info = fg_info()
        if info:
            g = safe_name(Path(info['process']).stem)
            if g != game or info['pid'] != pid:
                game, pid, session, seq = g, info['pid'], datetime.now().strftime('%Y%m%d_%H%M%S'), 0
                print(f'[SESSION] {game} | window={info["width"]}x{info["height"]} | {session}')
                print(f'[DISPLAY] monitor={info["monitor_left"]},{info["monitor_top"]}..{info["monitor_right"]},{info["monitor_bottom"]}')

            frame = None
            try:
                # Full-output grab avoids DXcam rejecting desktop-global window coordinates.
                frame = cam.grab()
            except Exception as e:
                print(f'[WARN] capture failed: {type(e).__name__}: {e}')
                try:
                    old_cam = cam
                    cam = None
                    try:
                        old_cam.stop()
                    except Exception:
                        pass
                    del old_cam
                    import gc
                    gc.collect()
                    cam = make_camera()
                    frame = cam.grab()
                except Exception as e2:
                    print(f'[WARN] camera recovery failed: {type(e2).__name__}: {e2}')

            if frame is not None:
                shape = (int(frame.shape[1]), int(frame.shape[0]))
                if shape != last_output_shape:
                    last_output_shape = shape
                    print(f'[DISPLAY] captured output={shape[0]}x{shape[1]}')
                cropped, coords, output_shape = crop_from_output(frame, info)
                if cropped is not None:
                    # Require the complete game client to be inside the selected output.
                    expected = (info['width'], info['height'])
                    actual = (int(cropped.shape[1]), int(cropped.shape[0]))
                    if actual != expected:
                        print(f'[SKIP] window crop is partial: requested={expected[0]}x{expected[1]} actual={actual[0]}x{actual[1]} relative={coords} output={output_shape}')
                    else:
                        seq += 1
                        ts = datetime.now()
                        fn = f'{seq:06d}_{ts:%H%M%S_%f}'[:-3] + '.png'
                        outdir = RAW_ROOT / game / session
                        outdir.mkdir(parents=True, exist_ok=True)
                        path = outdir / fn
                        Image.fromarray(cropped).save(
                            path,
                            format='PNG',
                            compress_level=int(CAP.get('png_compress_level', 3))
                        )
                        write_row({
                            'file': str(path.relative_to(RAW_ROOT)).replace('/', '\\'),
                            'game': game,
                            'session': session,
                            'timestamp': ts.isoformat(timespec='milliseconds'),
                            'process': info['process'],
                            'pid': info['pid'],
                            'window_title': info['title'],
                            'capture_width': int(cropped.shape[1]),
                            'capture_height': int(cropped.shape[0]),
                            'window_width': info['width'],
                            'window_height': info['height'],
                            'output_width': output_shape[0],
                            'output_height': output_shape[1],
                            'crop_x': coords[0],
                            'crop_y': coords[1],
                        })
                        print(f'[SAVE] {game}/{session}/{fn}')
        time.sleep(max(0.05, float(CAP['interval_seconds']) - (time.monotonic() - start)))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
