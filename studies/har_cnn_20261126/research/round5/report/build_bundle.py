#!/usr/bin/env python3
"""Package the frozen research artefacts without third-party papers or raw ZIPs."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[3]
R5=ROOT/'research/round5'
OUT=ROOT/'output/har_cnn_20261126'
DEST=OUT/'HAR_1D_CNN_复现包_20261126.zip'
def main():
    assert json.loads((R5/'cnn_experiments/audit_complete.json').read_text())['independent_sklearn_metrics_and_float64_nll_match']
    assert json.loads((R5/'motionsense_experiments/independent_audit.json').read_text())['passed']
    selected={}
    def add(p,arc=None):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':selected[arc or str(p.relative_to(ROOT))]=p
    for dirname in ['cnn_experiments','motionsense_experiments']:
        for p in (R5/dirname).rglob('*'):
            if p==R5/dirname/'data/train_val.npz':continue
            add(p)
    for p in (ROOT/'research/round3/experiments/cnn_pilot').rglob('*'):add(p)
    old=ROOT/'research/round3/experiments'
    for name in ['run_pilot.py','run_cnn_pilot.py']:add(old/name)
    for p in (old/'data').glob('*'):
        if p.name=='train_val.npz' or p.name.startswith('epoch_orders_seed'):add(p)
    for name in ['cnn_architecture_review.md','cnn_transfer_review.md','ls_kd_literature_addendum.md','harnet_weight_audit.json','check_harnet_weights.py','fetch_resources.py']:
        add(R5/name)
    add(ROOT/'research/round4/publication_review.md')
    for p in (R5/'report').glob('*'):
        if p.suffix in ['.py','.md']:add(p)
    for p in (R5/'sources').glob('*'):
        if p.name.startswith('ssl_') and p.suffix in ['.py','.md']:add(p)
    for p in OUT.iterdir():
        if p.suffix in ['.csv','.json','.png','.pdf','.md'] and p.name!='upload_result.json':add(p)
    add(R5/'README_bundle.md','README.md')
    manifest=[]
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for arc,p in sorted(selected.items()):
            data=p.read_bytes();z.writestr('HAR_1D_CNN_20261126/'+arc,data)
            manifest.append(dict(path=arc,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('HAR_1D_CNN_20261126/MANIFEST_SHA256.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
    print(json.dumps(dict(path=str(DEST),files=len(manifest)+1,bytes=DEST.stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
