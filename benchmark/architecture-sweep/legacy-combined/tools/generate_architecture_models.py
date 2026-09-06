from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

class Probe(nn.Module):
    def __init__(self, c:int, layers:int):
        super().__init__()
        if layers < 2: raise ValueError('layers must be >=2')
        seq=[nn.Conv2d(3,c,3,1,1), nn.ReLU(inplace=False)]
        for _ in range(layers-2): seq += [nn.Conv2d(c,c,3,1,1), nn.ReLU(inplace=False)]
        seq += [nn.Conv2d(c,3,3,1,1)]
        self.net=nn.Sequential(*seq)
    def forward(self,x): return self.net(x)

def params(c,l):
    if l==2: return (3*c*9+c)+(c*3*9+3)
    return (3*c*9+c)+(l-2)*(c*c*9+c)+(c*3*9+3)

def gmac(c,l,h,w):
    mac=(3*c*9)+(l-2)*(c*c*9)+(c*3*9)
    return h*w*mac/1e9

def make_calibration(dst:Path,w:int,h:int,n:int=8,seed:int=1234):
    dst.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    yy=np.linspace(0,1,h,dtype=np.float32)[:,None]
    xx=np.linspace(0,1,w,dtype=np.float32)[None,:]
    for i in range(n):
        noise=rng.random((h,w,3),dtype=np.float32)
        grad=np.stack([np.broadcast_to(xx,(h,w)),np.broadcast_to(yy,(h,w)),np.broadcast_to((xx+yy)/2,(h,w))],axis=2)
        arr=np.clip(0.55*noise+0.45*np.roll(grad,i%3,axis=2),0,1)
        Image.fromarray((arr*255).astype(np.uint8),'RGB').save(dst/f'cal_{i:02d}.png')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--channels',default='4,6,8,10,12,14,16,20,24,28,32')
    ap.add_argument('--layers',default='2,3,4,5,6,8')
    ap.add_argument('--width',type=int,default=1280); ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--cal-count',type=int,default=8); ap.add_argument('--skip-existing',action='store_true')
    args=ap.parse_args(); root=args.root.resolve(); models=root/'sweep_models'; floats=root/'float_models'; cal=root/'calibration_synthetic'
    models.mkdir(parents=True,exist_ok=True); floats.mkdir(parents=True,exist_ok=True)
    cs=[int(x) for x in args.channels.split(',') if x.strip()]; ls=[int(x) for x in args.layers.split(',') if x.strip()]
    make_calibration(cal,args.width,args.height,args.cal_count)
    manifest=[]; torch.manual_seed(1234)
    qscript=Path(__file__).with_name('int8_ptq.py')
    for c in cs:
      for l in ls:
        name=f'C{c}-L{l}'; outdir=models/name; outdir.mkdir(exist_ok=True)
        fp=floats/f'{name}.onnx'; qp=outdir/f'{name}-int8-qdq.onnx'
        if not (args.skip_existing and qp.exists()):
          model=Probe(c,l).eval(); x=torch.zeros(1,3,args.height,args.width,dtype=torch.float32)
          torch.onnx.export(model,x,str(fp),input_names=['input'],output_names=['residual'],opset_version=17,do_constant_folding=True)
          cmd=[sys.executable,str(qscript),'--model',str(fp),'--output',str(qp),'--calibration',str(cal),'--count',str(args.cal_count),'--width',str(args.width),'--height',str(args.height),'--require-core-only']
          print('[QUANT]',name,flush=True); subprocess.run(cmd,check=True)
        manifest.append({'architecture':name,'channels':c,'conv_layers':l,'parameters':params(c,l),'gmac_per_frame':round(gmac(c,l,args.height,args.width),4),'model':str(qp)})
    (root/'sweep_manifest.json').write_text(json.dumps({'schema_version':1,'input':[1,3,args.height,args.width],'channels':cs,'layers':ls,'models':manifest},indent=2),encoding='utf-8')
    with (root/'sweep_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
      w=csv.DictWriter(f,fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)
    print(f'[OK] Generated {len(manifest)} INT8 QDQ probe models')
if __name__=='__main__': main()
