"""本機 fixture regression：不連線 HPC，不執行正式矩陣。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def function_ast(file,name):
    tree=ast.parse(file.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();source=Path(args.source_root).resolve();report=Path(args.report).resolve()
    exact_files=['src/models.py','src/dlutils.py','src/pot.py','src/spot.py','src/constants.py','src/training_protocol.py']
    for file in exact_files:assert (ROOT/file).read_bytes()==(source/file).read_bytes(),file
    for file,names in [('main.py',['convert_to_windows','load_model','save_model','backprop']),('preprocess.py',['load_and_save','load_and_save2']),('src/cross_machine.py',['windows_for_machine','fit_source_threshold']),('src/experiment.py',['additional_metrics'])]:
        for name in names:assert function_ast(ROOT/file,name)==function_ast(source/file,name),(file,name)
    with tempfile.TemporaryDirectory(prefix='tranad-deploy-regression-') as temporary:
        tmp=Path(temporary);data=tmp/'SMD'
        for part in ['train','test','labels','interpretation_label']:(data/part).mkdir(parents=True)
        rng=np.random.default_rng(19)
        train=rng.uniform(.05,.25,(80,38));test=rng.uniform(.05,.25,(48,38));test[10:20,:2]=.9
        scalar=np.zeros(48,dtype=int);scalar[10:20]=1
        np.savetxt(data/'train/machine-1-1.txt',train,delimiter=',');np.savetxt(data/'test/machine-1-1.txt',test,delimiter=',')
        np.savetxt(data/'labels/machine-1-1.txt',scalar,fmt='%d');(data/'interpretation_label/machine-1-1.txt').write_text('10-20:1,2\n')
        for name,repo in [('reference',source),('deployment',ROOT)]:
            env=dict(os.environ,PROJECT_ROOT=str(repo),DATA_ROOT=str(data),RESULTS_ROOT=str(tmp/name),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(tmp/'mpl'))
            with (tmp/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'tests/reference_probe.py'),'--repo',str(repo),'--data',str(data),'--out',str(tmp/name)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((tmp/'reference/summary.json').read_text())==json.loads((tmp/'deployment/summary.json').read_text())
        for file in ['arrays.npz','state.npz','training_orders/epoch_0.npz']:
            with np.load(tmp/'reference'/file) as a,np.load(tmp/'deployment'/file) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        env=dict(os.environ,PROJECT_ROOT=str(ROOT),DATA_ROOT=str(data),RESULTS_ROOT=str(tmp/'cli-output'),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(tmp/'mpl'))
        dry=subprocess.check_output([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--hpc','--dry-run','--smoke-test','--epochs','1','--batch-size','8','--eval-batch-size','8'],env=env,text=True)
        contract=json.loads(dry);assert contract['window_count']==80 and contract['dtype']=='float64' and contract['protocol_version']=='v2'
        with (tmp/'cli-smoke.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--hpc','--smoke-test','--epochs','1','--batch-size','8','--eval-batch-size','8'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        run=next((tmp/'cli-output/runs').iterdir());assert (run/'evaluation.json').exists() and (run/'run.log').exists()
        missing=dict(env);missing.pop('DATA_ROOT')
        bad=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--hpc','--dry-run'],env=missing,capture_output=True,text=True)
        assert bad.returncode!=0 and 'DATA_ROOT' in bad.stderr
        result=dict(status='PASS',protected_source_bytes_equal=exact_files,
                    algorithm_function_asts_equal=True,training_loss_exact_equal=True,
                    trained_parameters_exact_equal=True,shuffle_order_and_generator_state_exact_equal=True,
                    labels_windows_scores_predictions_metrics_exact_equal=True,
                    standalone_cli_smoke='PASS',hpc_missing_data_root_rejected=True,
                    fixture=dict(train_points=80,test_points=48,features=38,epochs=1,batch_size=8),
                    fixture_metrics=json.loads((tmp/'deployment/summary.json').read_text()),
                    note='這是本機小型 fixture regression，不是 Hazel 執行或正式 experiment。')
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
