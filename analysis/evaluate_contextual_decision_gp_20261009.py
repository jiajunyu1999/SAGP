"""Freeze one validation-selected revision, then evaluate the original test cohorts."""
import argparse,json,datetime,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_contextual_decision_gp_20261009 import RESULTS,KINDS,feature_cache,cohorts,evaluate
from train_protected_decision_gp_20261009 import RESULTS as GLOBAL,evaluate as evaluate_global
from contextual_decision_gp_20261009 import ContextualDecisionGP
from protected_decision_gp_20261009 import ProtectedDecisionGP
from meta_gp_sar_20261007 import read,context_components,DEVICE,SEEDS
from controlled_joint_ablation_20261009 import RESULTS as CONTROL
from summarize_controlled_joint_ablation_20261009 import prepare,contrasts,pair_metrics
from prepare_sar_transfer_20261007 import digest


def select():
    complete=[json.loads((RESULTS/f'complete_{kind}_seed_{seed}.json').read_text()) for seed in SEEDS for kind in KINDS]
    for seed in SEEDS:
        group=[v for v in complete if v['seed']==seed]
        assert len({v['batch_sha256'] for v in group})==1
        old=json.loads((CONTROL/f'complete_joint_seed_{seed}.json').read_text())
        assert group[0]['batch_sha256']==old['batch_sha256']
    m=pd.concat([pd.read_csv(RESULTS/f'validation_metrics_{kind}_seed_{seed}.csv') for kind in KINDS for seed in SEEDS]+[pd.read_csv(GLOBAL/f'validation_metrics_detached_action_seed_{seed}.csv') for seed in SEEDS],ignore_index=True)
    m.to_csv(RESULTS/'all_validation_metrics.csv',index=False)
    scores=[]
    for kind in KINDS+['detached_action']:
        r=m.query('checkpoint=="selected" and model==@kind').groupby('task')[['AP','pair_macro_AUROC','macro_MAE']].mean()
        value=float(r[['AP','pair_macro_AUROC']].to_numpy().mean()-.1*r.macro_MAE.sum())
        scores.append({'model':kind,'score':value,**{f'task{t}_{key}':float(r.loc[t,key]) for t in [1,2] for key in ['AP','pair_macro_AUROC','macro_MAE']}})
    ranking=pd.DataFrame(scores).sort_values('score',ascending=False,kind='stable');ranking.to_csv(RESULTS/'validation_model_selection.csv',index=False)
    selection={'selected_model':ranking.iloc[0].model,'criterion':'mean four validation AP/within-pair AUROC minus0.1 sum two macroMAE, averaged across five seeds; one global model for both tasks',
        'validation_scores':scores,'frozen_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_seen_during_variant_selection':False,
        'benchmark_limit':'Earlier results from this benchmark informed the design. This is exploratory re-evaluation, not untouched external validation.',
        'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('train_contextual_decision_gp_20261009.py'),Path(__file__).with_name('contextual_decision_gp_20261009.py')]}}
    (RESULTS/'selection.json').write_text(json.dumps(selection,indent=2));print('FROZEN SELECTION',json.dumps(selection),flush=True)
    return selection


def main(select_only=False):
    torch.set_num_threads(4)
    if not (RESULTS/'selection.json').exists():selection=select()
    else:selection=json.loads((RESULTS/'selection.json').read_text())
    if select_only:return
    kind=selection['selected_model'];obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet')
    roots=context_components(obs);mapping,bits,desc=feature_cache();metrics=[];pair_rows=[]
    preds={}
    for seed in SEEDS:
        _,groups,_,_,_,frames=cohorts(seed,obs,data)
        for ck in ['selected','final']:
            cp=torch.load((GLOBAL if kind=='detached_action' else RESULTS)/f'{kind}_{ck}_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
            matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
            model=(ProtectedDecisionGP(cp['dim']) if kind=='detached_action' else ContextualDecisionGP(cp['dim'],kind)).to(DEVICE);model.load_state_dict(cp['state_dict']);model.eval()
            _,ms,p=(evaluate_global if kind=='detached_action' else evaluate)(model,frames['test'],groups,mapping,matrix)
            for t in [1,2]:
                path=RESULTS/f'task{t}_{kind}_{ck}_test_seed_{seed}.csv';p[t].to_csv(path,index=False);f=pd.read_csv(path)
                base=pd.read_csv(CONTROL/f'task{t}_joint_{ck}_test_seed_{seed}.csv')
                key='comparison_id' if t==1 else 'sample_id'
                assert all(np.array_equal(f[col].to_numpy(),base[col].to_numpy()) for col in [key,'pair_id','target_chembl_id','binary_loss'])
                assert np.allclose(f.target,base.target,rtol=0,atol=1e-8)
                metrics.append({'seed':seed,'task':t,'model':kind,'checkpoint':ck,'partition':'test','best_step':cp['best_step'],**ms[t]})
                pair_rows.append(pair_metrics(f).assign(seed=seed,task=t,checkpoint=ck,model=kind));preds[ck,t,seed]=f
            print('TEST',seed,ck,ms,flush=True)
    m=pd.DataFrame(metrics);m.to_csv(RESULTS/'test_metrics_by_seed.csv',index=False)
    pair_data=pd.concat(pair_rows,ignore_index=True);pair_data.to_csv(RESULTS/'test_metrics_by_seed_and_pair.csv',index=False)
    summary=m.groupby(['checkpoint','task'])[['AP','pair_macro_AUROC','macro_MAE']].agg(['mean','std']);summary.to_csv(RESULTS/'test_summary.csv')
    lookup=json.loads((CONTROL/'bootstrap_components.json').read_text());draws=np.load(CONTROL/'bootstrap_component_weights.npy')
    rows=[];differences=[]
    for ck in ['selected','final']:
        for task in [1,2]:
            for control in ['joint','task1_only' if task==1 else 'task2_only','shared_multihead']:
                packed=[]
                for seed in SEEDS:
                    a=preds[ck,task,seed];b=pd.read_csv(CONTROL/f'task{task}_{control}_{ck}_test_seed_{seed}.csv')
                    packed.append(prepare(a,b,roots,lookup))
                    pa=pair_metrics(a).set_index('pair_id');pb=pair_metrics(b).set_index('pair_id').loc[pa.index]
                    d=pa[['AP','pair_macro_AUROC','macro_MAE']]-pb[['AP','pair_macro_AUROC','macro_MAE']]
                    differences.append(d.reset_index().assign(seed=seed,task=task,checkpoint=ck,control=control))
                values=contrasts(packed,draws);np.save(RESULTS/f'bootstrap_task{task}_{ck}_minus_{control}.npy',values)
                for j,key in enumerate(['AP','pair_macro_AUROC','macro_MAE']):
                    valid=values[1:,j][np.isfinite(values[1:,j])];lo,hi=np.quantile(valid,[.025,.975])
                    rows.append({'checkpoint':ck,'task':task,'control':control,'metric':key,'new_minus_control':values[0,j],'CI_low':lo,'CI_high':hi,'evaluable_draws':len(valid)})
                print('BOOTSTRAP',ck,task,control,flush=True)
    ci=pd.DataFrame(rows);ci.to_csv(RESULTS/'paired_component_bootstrap.csv',index=False)
    pd.concat(differences,ignore_index=True).to_csv(RESULTS/'paired_assay_pair_differences.csv',index=False)
    lines=['# Contextual decision GP optimization','',f'Frozen validation-selected variant: **{kind}**. Ten new validation-only candidate fits plus the previously fitted five global-action controls; one model selected globally for both tasks before candidate test evaluation. All15 compared fits use the exact original episode hashes and1000updates.',
        '', 'One shared chemical GP predicts continuous signed shifts and joint covariance. Detached adapters calibrate SAR risk and sensitivity magnitude/risk; the global action has11 parameters, contextual linear56, contextual neural400. Classification gradients never change the GP or magnitude action. All adapters use detached GP moments; the contextual candidates also clip GP and decision gradients separately. Probabilities are discriminatively calibrated decisions, not Gaussian posterior event probabilities. No prediction ensemble.',
        '', 'The five-fold benchmark has already informed model design, so the test results are exploratory re-evaluation. Comparisons use the same2000 connected assay/document-component draws as the controlled ablation, with common weights across dependent seeds. All positive and negative differences are retained.',
        '', '|Checkpoint|Task|AP|Within-pair AUROC|Macro MAE|','|---|---|---|---|---|']
    for ck in ['selected','final']:
        for task in [1,2]:
            r=summary.loc[ck,task];lines.append('|'+ck+'|'+str(task)+'|'+'|'.join(f'{r[k,"mean"]:.4f} ± {r[k,"std"]:.4f}' for k in ['AP','pair_macro_AUROC','macro_MAE'])+'|')
    lines += ['','|Checkpoint|Task|Control|Metric|Difference|95% CI|','|---|---|---|---|---|---|']
    for r in ci.itertuples():lines.append(f'|{r.checkpoint}|{r.task}|{r.control}|{r.metric}|{r.new_minus_control:+.4f}|[{r.CI_low:+.4f}, {r.CI_high:+.4f}]|')
    (RESULTS/'REPORT.md').write_text('\n'.join(lines)+'\n');print(summary.to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--select-only',action='store_true');main(p.parse_args().select_only)
