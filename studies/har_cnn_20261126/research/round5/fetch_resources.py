#!/usr/bin/env python3
"""Fetch only the public inputs needed for optional reproduction; verify hashes."""
from pathlib import Path
import argparse,hashlib,json,urllib.request

HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def fetch(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert sha(path)==expected,f'Existing input checksum mismatch: {path}'
        print('Verified existing',path);return
    temp=path.with_suffix(path.suffix+'.part')
    urllib.request.urlretrieve(url,temp)
    assert sha(temp)==expected,f'Download checksum mismatch: {path}'
    temp.rename(path);print('Downloaded and verified',path)
def main():
    p=argparse.ArgumentParser();p.add_argument('resource',choices=['motionsense','harnet']);a=p.parse_args()
    if a.resource=='motionsense':
        m=json.loads((HERE/'motionsense_experiments/data/manifest.json').read_text())
        fetch(m['source_url'],HERE/'data/MotionSense_A_DeviceMotion.zip',m['source_sha256'])
    else:
        commit='150550ea5d41800229c95e36f88f5bf0d2e7cf04'
        fetch(f'https://raw.githubusercontent.com/OxWearables/ssl-wearables/{commit}/model_check_point/mtl_5_best.mdl',
            HERE/'sources/ssl_mtl_5_best.mdl','74ffaefbafb467b7253d121ef71fe734ee03bc802ba33c33a6aa1aef7f91de5e')
        fetch('https://wearables-files.ndph.ox.ac.uk/files/ssl/mtl_best.mdl',
            HERE/'sources/ssl_mtl_best.mdl','c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6')
if __name__=='__main__':main()
