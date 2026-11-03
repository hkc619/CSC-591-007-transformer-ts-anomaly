"""只讀外部完成結果，產生 formal CSV / gaps / 統計；缺 run 時明確停止。"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.paths import resolve_roots
from find_failed_runs import latest_status


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    for name in ['project-root','data-root','results-root']:parser.add_argument('--'+name)
    parser.add_argument('--hpc',action='store_true');args=parser.parse_args();roots=resolve_roots(args)
    statuses=latest_status(roots.results)
    with (ROOT/'configs/formal_28_machine_runs.csv').open() as f:expected=list(csv.DictReader(f))
    if any(i not in statuses or statuses[i]['exit_code']!=0 for i in range(56)):
        raise ValueError('56 runs 尚未全部成功；請先執行 find_failed_runs.py，不能產生假完整統計')
    rows=[]
    for row in expected:
        folder=Path(statuses[int(row['task_id'])]['run_path'])
        m=json.loads((folder/'metadata.json').read_text());e=json.loads((folder/'evaluation.json').read_text())
        assert not m['smoke_test'] and m['protocol']=='v2' and m['seed']==1
        assert m['source_machines']==json.loads(row['source_machines']) and m['target_machine']==row['target_machine']
        assert m['epochs']==5 and m['batch_size']==m['eval_batch_size']==128 and m['device']=='cpu' and m['dtype']=='float64'
        assert m['threshold_source']=='source_machine_training_scores' and m['pot_fit']['stream_score_count']==0
        if row['setting']=='cross_machine_zero_shot':assert not m['leakage_checks']['target_train_loaded']
        r=dict(target_machine=m['target_machine'],group=int(m['target_machine'].split('-')[1]),setting=m['setting'],source_machines=';'.join(m['source_machines']),seed=1)
        for key in ['precision','recall','f1']:r['raw_'+key]=e['raw'][key];r['pa_'+key]=e['point_adjusted'][key]
        r.update(pr_auc=e['continuous_score']['pr_auc_trapezoidal'],average_precision=e['continuous_score']['average_precision'],roc_auc=e['continuous_score']['roc_auc'],threshold=e['threshold'])
        for key in ['TP','TN','FP','FN']:r[key.lower()]=e['raw'][key]
        r.update(train_windows=m['train_windows'],test_points=m['test_timestamps'],train_time_sec=m['training_seconds'],inference_time_sec=m['test_inference_seconds'],checkpoint_size_bytes=m['checkpoint_size_bytes'],checkpoint_path=m['checkpoint'],epoch_losses=json.dumps(m['training_loss_per_epoch']))
        rows.append(r)
    gaps=[]
    for target in json.loads((ROOT/'configs/formal_protocol_v2.json').read_text())['machines']:
        per=next(r for r in rows if r['target_machine']==target and r['setting']=='per_machine')
        cross=next(r for r in rows if r['target_machine']==target and r['setting']=='cross_machine_zero_shot')
        g=dict(target_machine=target,group=per['group'],floor_effect_raw_f1=per['raw_f1']<.05)
        for metric in ['pr_auc','raw_f1','pa_f1']:
            g['per_machine_'+metric]=per[metric];g['cross_machine_'+metric]=cross[metric];g[metric+'_gap']=per[metric]-cross[metric]
        gaps.append(g)
    out=roots.results/'summaries';out.mkdir(exist_ok=True)
    write_csv(out/'formal_28_machine_results.csv',rows);write_csv(out/'formal_28_machine_gaps.csv',gaps)
    statistics=[]
    for metric in ['pr_auc','raw_f1']:
        for prefix in ['per_machine_','cross_machine_','']:
            key=prefix+metric if prefix else metric+'_gap';values=[g[key] for g in gaps]
            statistics.append(dict(metric=key,mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values,ddof=1)),min=float(min(values)),max=float(max(values)),iqr=float(np.quantile(values,.75)-np.quantile(values,.25))))
    write_csv(out/'formal_28_machine_statistics.csv',statistics)
    groups=[]
    for group in [1,2,3]:
        subset=[g for g in gaps if g['group']==group];r=dict(group=group,count=len(subset))
        for metric in ['pr_auc','raw_f1']:
            values=[g[metric+'_gap'] for g in subset];r['mean_'+metric+'_gap']=float(np.mean(values));r['median_'+metric+'_gap']=float(np.median(values))
        groups.append(r)
    write_csv(out/'formal_28_machine_groups.csv',groups)
    counts={metric:dict(degraded=sum(g[metric+'_gap']>0 for g in gaps),improved=sum(g[metric+'_gap']<0 for g in gaps),unchanged=sum(g[metric+'_gap']==0 for g in gaps)) for metric in ['pr_auc','raw_f1']}
    systems={setting:dict(evaluation_models=28,checkpoint_bytes=sum(r['checkpoint_size_bytes'] for r in rows if r['setting']==setting),training_seconds=sum(r['train_time_sec'] for r in rows if r['setting']==setting)) for setting in ['per_machine','cross_machine_zero_shot']}
    (out/'analysis.json').write_text(json.dumps(dict(counts=counts,systems=systems),indent=2))
    plots(gaps,groups,out/'figures')
    text='# 正式 28-machine 結果\n\n'+f"Cross-machine PR-AUC 在 {counts['pr_auc']['degraded']} / 28 台下降，{counts['pr_auc']['improved']} 台提升。\n\n"
    text+='全部 metrics、gap、group 與 aggregate statistics 見同目錄 CSV。std 使用 ddof=1，IQR=Q75−Q25。PR-AUC 使用 continuous score 的 trapezoidal area，AP 另存。PA 依賴真實 anomaly segments，不能取代 raw/ranking 結論。\n\n'
    text+='LOMO 的 28 shared models 是 evaluation count，不代表 production 必須部署 28 shared models。限制：single seed、TranAD only、source training scores fit POT、no target adaptation、natural source-size weighting、no balancing、CPU float64。\n'
    (out/'FORMAL_28_MACHINE_RESULTS.md').write_text(text)
    print('摘要已寫入：',out)

def plots(gaps,groups,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6))
    for group in [1,2,3]:
        subset=[g for g in gaps if g['group']==group]
        ax.scatter([g['per_machine_pr_auc'] for g in subset],[g['cross_machine_pr_auc'] for g in subset],label=f'Group {group}')
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Per-machine PR-AUC',ylabel='Cross-machine PR-AUC',xlim=(0,1),ylim=(0,1));ax.legend();save(fig,'pr_auc_scatter')
    for metric in ['pr_auc','raw_f1']:
        fig,ax=plt.subplots(figsize=(13,5));values=[g[metric+'_gap'] for g in gaps]
        ax.bar(range(28),values,color=['#35679a' if v>=0 else '#d98b36' for v in values]);ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(28),[g['target_machine'].replace('machine-','') for g in gaps],rotation=65);ax.set_ylabel(metric+' gap (per - cross)');save(fig,metric+'_gap_by_target')
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(values,bins='auto',edgecolor='white');ax.axvline(0,color='black',ls='--');ax.set(xlabel=metric+' gap',ylabel='Targets');save(fig,metric+'_gap_distribution')
    for metric in ['pr_auc','raw_f1']:
        fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([[g[metric+'_gap'] for g in gaps if g['group']==group] for group in [1,2,3]],tick_labels=['Group 1','Group 2','Group 3']);ax.axhline(0,color='black',ls='--');ax.set_ylabel(metric+' gap');save(fig,metric+'_group_comparison')

if __name__=='__main__':main()
