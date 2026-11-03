"""從 DATA_ROOT 列出資料、驗證全部 28 台，產生外部 dataset manifest。"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import resolve_roots,scalar_label_path


def run_rows(machines):
    rows=[]
    for setting in ['per_machine','cross_machine_zero_shot']:
        for target in machines:
            sources=[target] if setting=='per_machine' else [m for m in machines if m!=target]
            rows.append(dict(task_id=len(rows),setting=setting,target_machine=target,
                             source_machines=json.dumps(sources),protocol='v2',seed=1,epochs=5,batch_size=128,eval_batch_size=128))
    return rows


def main():
    parser=argparse.ArgumentParser()
    for name in ['project-root','data-root','results-root']:parser.add_argument('--'+name)
    parser.add_argument('--hpc',action='store_true');args=parser.parse_args();roots=resolve_roots(args)
    machine_sets={split:{p.stem for p in (roots.data/split).glob('machine-*.txt')} for split in ['train','test','interpretation_label']}
    machines=sorted(set.union(*machine_sets.values()),key=lambda s:tuple(map(int,s.split('-')[1:])))
    if len(machines)!=28 or any(set(machines)!=s for s in machine_sets.values()):raise ValueError('資料不完整或 machine 數不是 28；不會靜默跳過')
    info=[]
    for name in machines:
        train=np.genfromtxt(roots.data/'train'/(name+'.txt'),dtype=np.float64,delimiter=',')
        test=np.genfromtxt(roots.data/'test'/(name+'.txt'),dtype=np.float64,delimiter=',')
        assert train.ndim==test.ndim==2 and train.shape[1]==test.shape[1]==38
        assert np.isfinite(train).all() and np.isfinite(test).all()
        scalar=np.loadtxt(scalar_label_path(roots.data,name),dtype=int);assert scalar.shape==(len(test),)
        labels=np.zeros_like(test)
        for line in (roots.data/'interpretation_label'/(name+'.txt')).read_text().splitlines():
            span,cols=line.split(':');start,end=map(int,span.split('-'));features=[int(i)-1 for i in cols.split(',')]
            assert 0<=start<end<=len(test) and all(0<=i<38 for i in features)
            labels[start:end,features]=1
        union=labels.any(axis=1)
        info.append(dict(machine=name,group=int(name.split('-')[1]),train_length=len(train),test_length=len(test),feature_dimension=38,label_count=int(union.sum()),anomaly_ratio=float(union.mean()),scalar_disagreement_count=int(np.sum(union!=scalar))))
    expected=json.loads((ROOT/'configs/formal_protocol_v2.json').read_text())['machines']
    assert machines==expected,'DATA_ROOT machine list 與固定 config 不一致'
    with (ROOT/'configs/formal_28_machine_runs.csv').open() as f:assert list(csv.DictReader(f))==[{k:str(v) for k,v in row.items()} for row in run_rows(machines)]
    roots.results.mkdir(parents=True,exist_ok=True)
    path=roots.results/'machine_manifest.json';path.write_text(json.dumps(dict(machine_count=28,machines=info),indent=2))
    print('28 台資料驗證通過：',path)

if __name__=='__main__':main()
