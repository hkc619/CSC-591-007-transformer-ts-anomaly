"""在獨立 subprocess 比較來源與 deployment；所有輸出必須指向暫存目錄。"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
a=p.parse_args();repo=Path(a.repo).resolve();data=Path(a.data).resolve();out=Path(a.out).resolve();out.mkdir()
os.chdir(out);sys.path.insert(0,str(repo))
sys.argv=['reference-probe','--model','TranAD','--dataset','SMD','--protocol','v2','--epochs','1','--batch-size','8','--eval-batch-size','8']
if (repo/'configs/formal_protocol_v2.json').exists():sys.argv+=['--smoke-test']
os.environ.update(PROJECT_ROOT=str(repo),DATA_ROOT=str(data),RESULTS_ROOT=str(out))
import main as m
from src.training_protocol import seeded_generator
from src.cross_machine import fit_source_threshold
from src.experiment import additional_metrics
from src.pot import adjust_predicts,calc_point2point
import preprocess
preprocess.output_folder=str(out/'processed');(out/'processed/SMD').mkdir(parents=True)
name='machine-1-1'
preprocess.load_and_save('train',name+'.txt',name,str(data))
shape=preprocess.load_and_save('test',name+'.txt',name,str(data))
preprocess.load_and_save2('labels',name+'.txt',name,str(data),shape)
train=torch.from_numpy(np.load(out/f'processed/SMD/{name}_train.npy'))
test=torch.from_numpy(np.load(out/f'processed/SMD/{name}_test.npy'))
labels=np.load(out/f'processed/SMD/{name}_labels.npy')
random.seed(1);np.random.seed(1);torch.manual_seed(1)
m.args.training_generator=seeded_generator(1);m.args.output_dir=str(out)
m.args.progress=False;m.args.checkpoint=None
model,opt,scheduler,_,_=m.load_model('TranAD',38)
windows=m.convert_to_windows(train,model);tw=m.convert_to_windows(test,model)
loss,lr=m.backprop(0,model,windows,train,opt,scheduler)
model.eval();train_errors,_=m.backprop(0,model,windows,train,opt,scheduler,False)
errors,predictions=m.backprop(0,model,tw,test,opt,scheduler,False)
threshold,_=fit_source_threshold(train_errors.mean(axis=1));scores=errors.mean(axis=1);y=labels.any(axis=1).astype(int)
raw=scores>threshold;adjusted=adjust_predicts(scores,y,pred=raw.copy())
values=calc_point2point(adjusted,y);legacy=dict(zip(['f1','precision','recall','TP','TN','FP','FN','ROC/AUC'],values))
evaluation,_=additional_metrics(scores,y,threshold,legacy)
np.savez_compressed(out/'arrays.npz',windows=windows.numpy(),labels=labels,errors=errors,train_errors=train_errors,predictions=predictions,raw=raw,adjusted=adjusted)
np.savez_compressed(out/'state.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
(out/'summary.json').write_text(json.dumps(dict(loss=float(loss),lr=lr,threshold=threshold,evaluation=evaluation),indent=2))
