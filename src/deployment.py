"""Dry-run contract：不初始化模型、不寫 cache、不產生結果。"""
import json
from pathlib import Path
import numpy as np
from src.cross_machine import validate_protocol
from src.paths import scalar_label_path


def describe_experiment(args,roots):
    sources,target=validate_protocol(args)
    lengths={}
    for machine in sources:
        raw=np.genfromtxt(roots.data/'train'/(machine+'.txt'),dtype=np.float64,delimiter=',')
        assert raw.ndim==2 and raw.shape[1]==38 and np.isfinite(raw).all()
        lengths[machine]=len(raw)
    test=np.genfromtxt(roots.data/'test'/(target+'.txt'),dtype=np.float64,delimiter=',')
    assert test.ndim==2 and test.shape[1]==38 and np.isfinite(test).all()
    if not (roots.data/'interpretation_label'/(target+'.txt')).is_file():raise FileNotFoundError('缺少 target interpretation_label')
    scalar_label_path(roots.data,target)
    config=json.loads((roots.project/'configs/formal_protocol_v2.json').read_text())
    return dict(sources=sources,target=target,source_lengths=lengths,window_count=sum(lengths.values()),
                test_points=len(test),protocol_version='v2',pot_mode=config['pot']['mode'],
                pot_parameters=config['pot'],seed=args.seed,model='TranAD',feature_dimension=38,
                window=10,learning_rate=.0001,epochs=args.epochs,batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,dtype='float64',device='cpu',
                shuffle=True,machine_balanced_sampling=False,
                model_configuration=dict(d_model=76,nhead=38,dim_feedforward=16,window=10),
                project_root=str(roots.project),data_root=str(roots.data),results_root=str(roots.results),
                smoke_test=args.smoke_test)
