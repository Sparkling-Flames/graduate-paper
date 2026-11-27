#!/usr/bin/env python3
"""Independent metric/protocol audit from saved MotionSense outputs, no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE=Path(__file__).resolve().parent
assert (HERE/'completion.json').exists()
manifest=json.loads((HERE/'data/manifest.json').read_text())
with np.load(HERE/'data/train_val.npz') as data:
    truth=data['y_val'];subjects=data['subject_val']
sealed=set(manifest['sealed_entries_not_opened'])
assert not sealed.intersection(manifest['opened_entries'])
assert len(sealed)==60
max_error={'macro_f1':0.,'accuracy':0.,'nll':0.};checks=0
for p in sorted((HERE/'runs').glob('*/result.json')):
    r=json.loads(p.read_text());curve=pd.read_csv(p.parent/'curves.csv')
    assert not r['test_contents_opened']
    for snap,stats in r['snapshots'].items():
        with np.load(p.parent/f'{snap}_predictions.npz') as z:
            assert np.array_equal(z['y'],truth) and np.array_equal(z['subject'],subjects)
            logit=z['logits'].astype(np.float64)
        pred=logit.argmax(1)
        recomputed=dict(accuracy=float(accuracy_score(truth,pred)),
            macro_f1=float(f1_score(truth,pred,average='macro',labels=range(6),zero_division=0)),
            nll=float((logsumexp(logit,axis=1)-logit[np.arange(len(truth)),truth]).mean()))
        for key,value in recomputed.items():
            err=abs(value-stats['validation'][key]);max_error[key]=max(max_error[key],err)
            assert err < (2e-6 if key=='nll' else 1e-12),(p,snap,key,err)
        assert np.array_equal(confusion_matrix(truth,pred,labels=range(6)),stats['validation']['confusion'])
        for s in sorted(set(subjects)):
            keep=subjects==s
            score=f1_score(truth[keep],pred[keep],average='macro',labels=range(6),zero_division=0)
            assert abs(score-stats['per_subject'][str(s)]['macro_f1'])<1e-12
        budget=int(snap.replace('best','').replace('last',''))
        if snap.startswith('best'):
            eligible=curve[curve.epoch<=budget]
            assert abs(stats['validation']['macro_f1']-eligible.val_macro_f1.max())<1e-12
            selected=eligible[eligible.epoch==stats['epoch']].iloc[0]
            assert abs(selected.val_macro_f1-stats['validation']['macro_f1'])<1e-12
        checks+=1
for seed in [20261121,20261122,20261123]:
    runs=[json.loads((HERE/'runs'/f'student_w8_{m}_seed{seed}'/'result.json').read_text()) for m in ['ce','ls','kd']]
    assert len({x['initial_state_sha256'] for x in runs})==1
    assert len({x['orders_sha256'] for x in runs})==1
out=dict(checkpoints_checked=checks,independent_metric_max_errors=max_error,
    paired_initialization_orders=True,confusions_and_per_subject_metrics_verified=True,
    max_f1_checkpoint_rule_verified=True,sealed_test_entries_not_read=60,passed=True)
(HERE/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
