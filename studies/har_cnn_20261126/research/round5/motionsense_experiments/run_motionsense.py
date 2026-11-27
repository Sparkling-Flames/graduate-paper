#!/usr/bin/env python3
"""Fixed-protocol MotionSense development-only temporal CNN experiment."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, copy, hashlib, io, json, platform, random, re, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'data' / 'MotionSense_A_DeviceMotion.zip'
TRAIN = [1,4,6,7,8,10,11,12,13,15,16,17,20,21,23,24]
VAL = [2,5,14,19]
TEST = [3,9,18,22]
CLASSES = ['downstairs','upstairs','walking','jogging','sitting','standing']
PREFIXES = ['dws','ups','wlk','jog','sit','std']
CHANNELS = [f'{signal}.{axis}' for signal in ['userAcceleration','rotationRate','gravity'] for axis in ['x','y','z']]
CONFIG = dict(protocol='motionsense_cnn_v1', train_subjects=TRAIN, val_subjects=VAL,
    test_subjects_sealed=TEST, channels=CHANNELS, class_names=CLASSES, window=128, stride=64,
    nominal_hz=50, teacher_width=32, student_width=8, teacher_seed=20261120,
    seeds=[20261121,20261122,20261123], teacher_epochs=80, student_epochs=160,
    lr=.001, batch_size=128, epsilon=.1, temperature=2., teacher_weight=.5,
    threads=2, test_contents_opened=False)

def now(): return datetime.now(timezone.utc).isoformat()
def writej(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def state_hash(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

class CNN(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(9,w,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(w,2*w,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.classifier=nn.Linear(2*w,6)
    def forward(self,x): return self.classifier(self.features(x).squeeze(-1))

@torch.inference_mode()
def predict(model,x):
    model.eval()
    return torch.cat([model(x[i:i+1024]) for i in range(0,len(x),1024)])

def metrics(logits,y):
    truth=y.numpy();pred=logits.argmax(1).numpy()
    cm=np.bincount(truth*6+pred,minlength=36).reshape(6,6)
    tp=cm.diagonal().astype(float);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(6),where=den>0)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),
        nll=float(F.cross_entropy(logits,y)),confusion=cm.tolist(),class_f1=f1.tolist())

def prepare():
    out=HERE/'data';out.mkdir(exist_ok=True)
    if (out/'train_val.npz').exists():
        m=json.loads((out/'manifest.json').read_text())
        assert m['train_subjects']==TRAIN and m['val_subjects']==VAL and m['channels']==CHANNELS
        return
    assert set(TRAIN).isdisjoint(VAL) and set(TRAIN+VAL).isdisjoint(TEST)
    blocks={'train':[], 'val':[]};ys={'train':[], 'val':[]};rows=[];opened=[];sealed=[];trials=[]
    with zipfile.ZipFile(SOURCE) as z:
        for name in sorted(z.namelist()):
            m=re.fullmatch(r'A_DeviceMotion_data/([a-z]+)_(\d+)/sub_(\d+)\.csv',name)
            if not m:continue
            prefix,trial,sub=m.group(1),int(m.group(2)),int(m.group(3))
            assert prefix in PREFIXES and sub in TRAIN+VAL+TEST
            if sub in TEST:
                sealed.append(name);continue
            split='train' if sub in TRAIN else 'val'
            opened.append(name)
            frame=pd.read_csv(io.BytesIO(z.read(name)),usecols=CHANNELS)
            a=frame[CHANNELS].to_numpy(dtype=np.float32)
            assert np.isfinite(a).all() and a.shape[1]==9
            label=PREFIXES.index(prefix)
            starts=list(range(0,len(a)-128+1,64))
            trials.append(dict(subject=sub,trial=trial,activity=CLASSES[label],split=split,
                samples=len(a),windows=len(starts),source_entry=name))
            for start in starts:
                blocks[split].append(a[start:start+128].T)
                ys[split].append(label)
                rows.append(dict(split=split,split_row=len(ys[split])-1,subject=sub,trial=trial,
                    label=label,start_sample=start,end_sample_exclusive=start+128,source_entry=name))
    assert len(opened)==20*15 and len(sealed)==4*15
    x={k:np.stack(v) for k,v in blocks.items()}
    mean=x['train'].mean((0,2),dtype=np.float64,keepdims=True)
    std=x['train'].std((0,2),dtype=np.float64,keepdims=True);std[std==0]=1
    meta=pd.DataFrame(rows)
    cached={}
    for split in ['train','val']:
        cached['x_'+split]=np.ascontiguousarray((x[split]-mean)/std,dtype=np.float32)
        cached['y_'+split]=np.array(ys[split],dtype=np.int64)
        cached['subject_'+split]=meta[meta.split==split].sort_values('split_row').subject.to_numpy(dtype=np.int64)
        assert set(cached['y_'+split])==set(range(6))
    np.savez_compressed(out/'train_val.npz',**cached)
    np.savez(out/'channel_scaler_train_only.npz',mean=mean,std=std)
    meta.to_csv(out/'window_manifest.csv',index=False)
    pd.DataFrame(trials).to_csv(out/'trial_manifest.csv',index=False)
    meta.groupby(['split','subject','label']).size().rename('windows').reset_index().to_csv(out/'subject_class_counts.csv',index=False)
    writej(out/'manifest.json',dict(source_url='https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip',
        source_bytes=SOURCE.stat().st_size,source_sha256=digest(SOURCE),train_subjects=TRAIN,val_subjects=VAL,
        test_subjects_sealed=TEST,opened_entries=opened,sealed_entries_not_opened=sealed,channels=CHANNELS,
        class_names=CLASSES,train_shape=list(cached['x_train'].shape),val_shape=list(cached['x_val'].shape),
        normalization_mean=mean.flatten().tolist(),normalization_std=std.flatten().tolist(),
        cache_sha256=digest(out/'train_val.npz'),test_contents_opened=False,prepared_utc=now()))
    print(json.dumps(dict(event='data_prepared',train=cached['x_train'].shape,val=cached['x_val'].shape,test_opened=False)),flush=True)

def save_snapshot(path,model,data,budget,kind,epoch):
    logits=predict(model,data['xv']);train_logits=predict(model,data['xt'])
    state=model.state_dict();torch.save(state,path/f'{kind}{budget}_state.pt')
    result=dict(epoch=epoch,train=metrics(train_logits,data['yt']),validation=metrics(logits,data['yv']),
        per_subject={str(s):metrics(logits[data['sv']==s],data['yv'][data['sv']==s]) for s in VAL},
        state_sha256=state_hash(state))
    np.savez_compressed(path/f'{kind}{budget}_predictions.npz',logits=logits.numpy(),
        y=data['yv'].numpy(),subject=data['sv'])
    return result

def run(name,w,seed,method,epochs,data,teacher_logits=None,teacher_sha=None):
    path=HERE/'runs'/name;path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists():return json.loads((path/'result.json').read_text())
    seed_all(seed);model=CNN(w);initial_sha=state_hash(model.state_dict())
    torch.save(model.state_dict(),path/'initial_state.pt')
    orders_path=HERE/'data'/f'orders_{seed}_{epochs}.npy'
    if not orders_path.exists():
        rng=np.random.default_rng(seed+1000000)
        np.save(orders_path,np.stack([rng.permutation(len(data['yt'])) for _ in range(epochs)]).astype(np.int64))
    orders=np.load(orders_path);assert orders.shape==(epochs,len(data['yt']))
    order_sha=digest(orders_path)
    writej(path/'config.json',CONFIG|dict(name=name,width=w,seed=seed,method=method,epochs=epochs,
        initial_state_sha256=initial_sha,orders_sha256=order_sha,teacher_state_sha256=teacher_sha,started_utc=now()))
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    best_key=None;best_state=None;best_epoch=None;curves=[];snapshots={};started=time.perf_counter()
    for epoch,order in enumerate(orders,start=1):
        model.train();total=0.;epoch_start=time.perf_counter()
        for ids in torch.from_numpy(order).split(128):
            opt.zero_grad(set_to_none=True);logits=model(data['xt'][ids]);labels=data['yt'][ids]
            hard=F.cross_entropy(logits,labels)
            if method=='ce':loss=hard
            elif method=='ls':loss=F.cross_entropy(logits,labels,label_smoothing=.1)
            elif method=='kd':
                t=teacher_logits[ids].detach()
                loss=.5*hard+.5*4*F.kl_div(F.log_softmax(logits/2,1),F.softmax(t/2,1),reduction='batchmean')
            else:raise ValueError(method)
            assert torch.isfinite(loss)
            loss.backward();opt.step();total+=float(loss.detach())*len(ids)
        tr=metrics(predict(model,data['xt']),data['yt']);va=metrics(predict(model,data['xv']),data['yv'])
        curves.append(dict(epoch=epoch,objective_loss=total/len(data['yt']),train_accuracy=tr['accuracy'],
            train_macro_f1=tr['macro_f1'],train_nll=tr['nll'],val_accuracy=va['accuracy'],
            val_macro_f1=va['macro_f1'],val_nll=va['nll'],epoch_seconds=time.perf_counter()-epoch_start))
        key=(va['macro_f1'],-va['nll'])
        if best_key is None or key>best_key:best_key=key;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        if epoch==5:
            estimate=(time.perf_counter()-started)/5*epochs
            print(json.dumps(dict(event='cost_probe',run=name,estimated_seconds=estimate)),flush=True)
            if name.startswith('teacher') and estimate>600:
                pd.DataFrame(curves).to_csv(path/'curves.csv',index=False)
                writej(HERE/'stopped.json',dict(reason='prespecified_teacher_cost_limit',estimated_seconds=estimate,limit=600))
                raise RuntimeError('Prespecified cost threshold reached')
        if epoch in [80,160]:
            last_state=copy.deepcopy(model.state_dict())
            snapshots[f'last{epoch}']=save_snapshot(path,model,data,epoch,'last',epoch)
            model.load_state_dict(best_state)
            snapshots[f'best{epoch}']=save_snapshot(path,model,data,epoch,'best',best_epoch)
            model.load_state_dict(last_state)
        if epoch%10==0:pd.DataFrame(curves).to_csv(path/'curves.csv',index=False)
    result=dict(name=name,width=w,seed=seed,method=method,epochs=epochs,snapshots=snapshots,
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-started,
        initial_state_sha256=initial_sha,orders_sha256=order_sha,teacher_state_sha256=teacher_sha,
        test_contents_opened=False,completed_utc=now())
    writej(path/'result.json',result)
    print(json.dumps(dict(event='completed',run=name,best80_f1=snapshots['best80']['validation']['macro_f1'],
        best160_f1=snapshots.get('best160',{}).get('validation',{}).get('macro_f1'),seconds=result['training_seconds'])),flush=True)
    return result

def summarize():
    rows=[]
    for p in sorted((HERE/'runs').glob('*/result.json')):
        r=json.loads(p.read_text())
        for snap,m in r['snapshots'].items():
            rows.append(dict(run=r['name'],width=r['width'],seed=r['seed'],method=r['method'],snapshot=snap,
                epoch=m['epoch'],parameters=r['parameters'],train_accuracy=m['train']['accuracy'],
                train_macro_f1=m['train']['macro_f1'],train_nll=m['train']['nll'],
                val_accuracy=m['validation']['accuracy'],val_macro_f1=m['validation']['macro_f1'],val_nll=m['validation']['nll']))
    f=pd.DataFrame(rows);f.to_csv(HERE/'summary_runs.csv',index=False)
    if len(f):
        s=f[f.width==8]
        s.groupby(['snapshot','method']).agg(n=('seed','size'),mean_f1=('val_macro_f1','mean'),sd_f1=('val_macro_f1','std')).to_csv(HERE/'summary_methods.csv')
        pairs=[]
        for snap in s.snapshot.unique():
            z=s[s.snapshot==snap].pivot(index='seed',columns='method',values='val_macro_f1')
            for a,b in [('kd','ce'),('kd','ls'),('ls','ce')]:
                if a in z and b in z:
                    for seed,v in ((z[a]-z[b])*100).items():pairs.append(dict(snapshot=snap,seed=seed,contrast=f'{a}-{b}',delta_pp=v))
        pd.DataFrame(pairs).to_csv(HERE/'paired_differences.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','summarize']);args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if (HERE/'config.json').exists():assert json.loads((HERE/'config.json').read_text())==CONFIG
    else:writej(HERE/'config.json',CONFIG)
    if args.action=='summarize':summarize();return
    prepare()
    if args.action=='prepare':return
    writej(HERE/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        pandas=pd.__version__,platform=platform.platform(),threads=torch.get_num_threads(),
        deterministic_algorithms=True,script_sha256=digest(Path(__file__)),protocol_sha256=digest(HERE/'protocol.md'),started_utc=now()))
    with np.load(HERE/'data'/'train_val.npz') as a:
        data=dict(xt=torch.from_numpy(a['x_train']),yt=torch.from_numpy(a['y_train']),
            xv=torch.from_numpy(a['x_val']),yv=torch.from_numpy(a['y_val']),sv=a['subject_val'])
    teacher_name='teacher_ce_seed20261120'
    t=run(teacher_name,32,20261120,'ce',80,data)
    teacher=CNN(32);teacher.load_state_dict(torch.load(HERE/'runs'/teacher_name/'best80_state.pt',weights_only=True))
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    t_sha=state_hash(teacher.state_dict());t_logits=predict(teacher,data['xt']).detach()
    torch.save(t_logits,HERE/'data'/'teacher_train_logits.pt')
    for seed in CONFIG['seeds']:
        results=[]
        for method in ['ce','ls','kd']:
            results.append(run(f'student_w8_{method}_seed{seed}',8,seed,method,160,data,
                t_logits if method=='kd' else None,t_sha if method=='kd' else None))
            summarize()
        assert len({r['initial_state_sha256'] for r in results})==1
        assert len({r['orders_sha256'] for r in results})==1
    assert t_sha==state_hash(teacher.state_dict()) and not t_logits.requires_grad
    writej(HERE/'completion.json',dict(new_models=10,paired_initialization_order_verified=True,
        frozen_teacher_verified=True,test_contents_opened=False,completed_utc=now()))
    summarize()

if __name__=='__main__':main()
