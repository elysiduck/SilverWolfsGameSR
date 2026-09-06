from __future__ import annotations
import time
from pathlib import Path
import dxcam
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'capture_test.png'

print('GameSR capture test')
print('This will capture the primary monitor once and save capture_test.png.')
print('Press Ctrl+C to cancel.')

time.sleep(2)
cam = dxcam.create(output_idx=0, output_color='RGB', processor_backend='numpy')
frame = cam.grab()
if frame is None:
    raise RuntimeError('No frame was returned by DXcam.')

Image.fromarray(frame).save(OUT, format='PNG', compress_level=1)
print(f'OK: {OUT}')
print(f'Size: {frame.shape[1]} x {frame.shape[0]}')
