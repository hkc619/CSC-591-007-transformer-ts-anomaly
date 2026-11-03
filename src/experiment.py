"""外部化 run bookkeeping；additional_metrics 函式保持 validated source 原文。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from uuid import uuid4
import numpy as np
import torch
from sklearn.metrics import average_precision_score,auc,precision_recall_curve,roc_auc_score
from src.paths import resolve_roots


def prepare_run(args):
    roots=resolve_roots()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    from src.training_protocol import seeded_generator
    args.training_generator=seeded_generator(args.seed)
    sources=args.train_machines or [args.machine]
    target=args.test_machine if args.train_machines else args.machine
    setting='cross_machine_zero_shot' if args.train_machines else 'per_machine'
    kind='smoke' if args.smoke_test else 'formal'
    name=f'{kind}_TranAD_SMD_{target}_v2_e{args.epochs}_bs{args.batch_size}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}'
    parent=Path(args.output_dir).expanduser().resolve() if args.output_dir else roots.results/'runs'
    if not parent.is_relative_to(roots.results):raise ValueError('--output-dir 必須位於 RESULTS_ROOT 內')
    folder=parent/name;folder.mkdir(parents=True,exist_ok=False)
    args.output_dir=str(folder);args.run_id=name
    args.save_checkpoint=str(roots.results/'checkpoints'/name/'model.ckpt')
    import preprocess
    preprocess.output_folder=str(folder/'cache/processed')
    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','matplotlib','tqdm']}
    provenance=json.loads((roots.project/'configs/source_provenance.json').read_text())
    commit=subprocess.run(['git','-C',str(roots.project),'rev-parse','HEAD'],capture_output=True,text=True)
    meta=dict(command=[sys.executable,*sys.argv],cwd=os.getcwd(),python=sys.version,platform=platform.platform(),
              packages=versions,seed=args.seed,device='cpu',dtype='float64',machine=target,target_machine=target,
              source_machines=sources,setting=setting,protocol='v2',training_shuffle=True,sampling='natural_source_size',
              epochs=args.epochs,batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
              checkpoint=args.save_checkpoint,source_checkpoint=None,window=10,torch_threads=torch.get_num_threads(),
              smoke_test=args.smoke_test,project_root=str(roots.project),data_root=str(roots.data),results_root=str(roots.results),
              deployment_git_commit=commit.stdout.strip() if commit.returncode==0 else None,
              source_git_commit=provenance['source_git_commit'],source_was_dirty=provenance['source_dirty'],
              data_sha256={str(roots.data/'train'/(m+'.txt')):hashlib.sha256((roots.data/'train'/(m+'.txt')).read_bytes()).hexdigest() for m in sources})
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    snapshot=folder/'source'
    for file in ['main.py','preprocess.py',*[str(p.relative_to(roots.project)) for p in (roots.project/'src').glob('*.py')], 'configs/formal_protocol_v2.json','configs/source_provenance.json']:
        dest=snapshot/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(roots.project/file,dest)
    print('Run directory:',folder,flush=True)
    return meta

def additional_metrics(scores, labels, threshold, legacy):
    raw = scores > threshold
    tp = int(np.sum(raw & (labels == 1)))
    fp = int(np.sum(raw & (labels == 0)))
    fn = int(np.sum(~raw & (labels == 1)))
    tn = int(np.sum(~raw & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    p, r, _ = precision_recall_curve(labels, scores)
    result = dict(threshold=float(threshold),
                  point_adjusted={k:float(legacy[k]) for k in ['precision','recall','f1','TP','TN','FP','FN']},
                  raw=dict(precision=precision,recall=recall,f1=f1,TP=tp,TN=tn,FP=fp,FN=fn),
                  continuous_score=dict(pr_auc_trapezoidal=float(auc(r,p)),
                                        average_precision=float(average_precision_score(labels,scores)),
                                        roc_auc=float(roc_auc_score(labels,scores))))
    return result, raw
