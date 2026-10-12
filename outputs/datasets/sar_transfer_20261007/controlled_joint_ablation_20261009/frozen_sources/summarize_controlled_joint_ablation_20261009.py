"""All-seed, all-pair differences with shared dependency-component bootstrap."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from controlled_joint_ablation_20261009 import RESULTS,MODELS,SEEDS
from meta_gp_sar_20261007 import context_components,read

METRICS=['AP','pair_macro_AUROC','macro_MAE']


def ap_cache(y,p,component):
    order=np.argsort(-p,kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(p[order])!=0),len(p)-1]
    return y[order],component[order],ends


def weighted_ap(cache,w):
    y,c,ends=cache;weights=w[c]
    tp=np.cumsum(weights*y)[ends];total=np.cumsum(weights)[ends]
    if tp[-1]<=0:return np.nan
    precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    return float(np.dot(np.diff(np.r_[0,tp]),precision)/tp[-1])


def pair_metrics(f):
    rows=[]
    for pair,g in f.groupby('pair_id',sort=False):
        y=g.binary_loss.to_numpy();valid=len(g)>=10 and y.sum()>=3 and (1-y).sum()>=3
        rows.append({'pair_id':pair,'n':len(g),'positive':int(y.sum()),'eligible_AUROC':valid,
                     'AP':float(average_precision_score(y,g.prediction)) if y.sum()>0 else np.nan,
                     'pair_macro_AUROC':float(roc_auc_score(y,g.prediction)) if valid else np.nan,
                     'macro_MAE':float(abs(g.target-g.regression).mean())})
    return pd.DataFrame(rows)


def prepare(a,b,roots,lookup):
    key='comparison_id' if 'comparison_id' in a else 'sample_id'
    # Parquet uses int8/bool labels; serialized predictions use int64. Match
    # exact values and order without mistaking a dtype change for new labels.
    assert all(np.array_equal(a[col].to_numpy(),b[col].to_numpy()) for col in [key,'pair_id','binary_loss'])
    assert np.allclose(a.target,b.target,rtol=0,atol=1e-8)
    c=np.array([lookup[roots[p]] for p in a.pair_id]);y=a.binary_loss.to_numpy()
    pa=pair_metrics(a).set_index('pair_id');pb=pair_metrics(b).set_index('pair_id').loc[pa.index]
    pc=np.array([lookup[roots[p]] for p in pa.index]);valid=pa.eligible_AUROC.to_numpy()
    return (ap_cache(y,a.prediction.to_numpy(),c),ap_cache(y,b.prediction.to_numpy(),c),pc[valid],
            (pa.pair_macro_AUROC-pb.pair_macro_AUROC).to_numpy()[valid],pc,(pa.macro_MAE-pb.macro_MAE).to_numpy())


def contrasts(packed,draws):
    out=np.full((len(draws),3),np.nan)
    for i,w in enumerate(draws):
        values=[]
        for aa,bb,pc,delta,ec,errors in packed:
            auc=np.average(delta,weights=w[pc]) if w[pc].sum()>0 else np.nan
            mae=np.average(errors,weights=w[ec]) if w[ec].sum()>0 else np.nan
            values.append([weighted_ap(aa,w)-weighted_ap(bb,w),auc,mae])
        vals=np.array(values)
        # Retain the same five-seed estimand; do not drop unevaluable seeds.
        good=np.isfinite(vals).all(0)
        out[i,good]=vals[:,good].mean(0)
    return out


def main(reps):
    all_done=[RESULTS/f'complete_{m}_seed_{s}.json' for s in SEEDS for m in MODELS]
    assert all(p.exists() for p in all_done),'All 35 fits must finish before aggregation.'
    completed=[json.loads(p.read_text()) for p in all_done]
    for seed in SEEDS:
        runs=[r for r in completed if r['seed']==seed]
        assert len({r['batch_sha256'] for r in runs})==1,'Episode streams differ!'
        assert len({r['steps'] for r in runs})==1
    pd.DataFrame(completed).to_csv(RESULTS/'compute_and_control_audit.csv',index=False)
    ms=pd.concat([pd.read_csv(RESULTS/f'metrics_{m}_seed_{s}.csv') for s in SEEDS for m in MODELS],ignore_index=True)
    ms.to_csv(RESULTS/'metrics_by_seed.csv',index=False)
    summary=ms.query('partition=="test"').groupby(['checkpoint','task','model'])[METRICS].agg(['mean','std'])
    summary.to_csv(RESULTS/'test_summary.csv')
    roots=context_components(read('molecule_assay_observations.parquet'))
    predictions={};pair_rows=[];components=set()
    for ck in ['selected','final']:
        for task in [1,2]:
            for seed in SEEDS:
                for model in MODELS:
                    p=RESULTS/f'task{task}_{model}_{ck}_test_seed_{seed}.csv';f=pd.read_csv(p)
                    components.update(roots[pair] for pair in f.pair_id)
                    predictions[ck,task,seed,model]=f
                    pair_rows.append(pair_metrics(f).assign(checkpoint=ck,task=task,seed=seed,model=model))
    pairs=pd.concat(pair_rows,ignore_index=True);pairs.to_csv(RESULTS/'metrics_by_seed_and_pair.csv',index=False)
    base=pairs[pairs.model.eq('joint')].rename(columns={m:m+'_joint' for m in METRICS})
    delta=pairs[pairs.model.ne('joint')].merge(base[['checkpoint','task','seed','pair_id']+[m+'_joint' for m in METRICS]],on=['checkpoint','task','seed','pair_id'],validate='many_to_one')
    for m in METRICS:delta[m+'_joint_minus_control']=delta[m+'_joint']-delta[m]
    delta.to_csv(RESULTS/'paired_assay_pair_differences.csv',index=False)
    lookup={c:i for i,c in enumerate(sorted(components))};rng=np.random.default_rng(20261009)
    draws=np.vstack([np.ones(len(lookup)),rng.multinomial(len(lookup),np.full(len(lookup),1/len(lookup)),size=reps)])
    np.save(RESULTS/'bootstrap_component_weights.npy',draws.astype(np.int32))
    (RESULTS/'bootstrap_components.json').write_text(json.dumps(lookup,indent=2))
    results=[]
    for ck in ['selected','final']:
        for task in [1,2]:
            for model in [m for m in MODELS if m!='joint']:
                packed=[prepare(predictions[ck,task,s,'joint'],predictions[ck,task,s,model],roots,lookup) for s in SEEDS]
                values=contrasts(packed,draws)
                np.save(RESULTS/f'bootstrap_task{task}_{ck}_joint_minus_{model}.npy',values)
                for j,m in enumerate(METRICS):
                    finite=np.isfinite(values[1:,j]);lo,hi=np.quantile(values[1:,j][finite],[.025,.975])
                    seed_ms=ms.query('partition=="test" and checkpoint==@ck and task==@task').pivot(index='seed',columns='model',values=m)
                    sd=seed_ms['joint']-seed_ms[model];assert np.isclose(values[0,j],sd.mean(),atol=1e-8)
                    results.append({'checkpoint':ck,'task':task,'control':model,'metric':m,'joint_minus_control':values[0,j],
                                    'CI_low':lo,'CI_high':hi,'evaluable_bootstrap_draws':int(finite.sum()),'requested_draws':reps,
                                    'seed_difference_SD':float(sd.std()),'seeds_joint_better':int((sd<0).sum() if m=='macro_MAE' else (sd>0).sum()),
                                    'source_components':len(lookup)})
                print('BOOTSTRAP',ck,task,model,flush=True)
    ci=pd.DataFrame(results);ci.to_csv(RESULTS/'paired_component_bootstrap.csv',index=False)
    lines=['# Controlled Joint GP ablation','',
           'All five original scaffold splits, identical query/support episode streams, identical test cohorts, root-supervision warm starts and preprocessing. Each variant completes 1,000 updates; checkpoint selection uses the same frozen validation rule. Final-step results test dependence on checkpoint selection.',
           '', 'A/B vs C tests auxiliary-supervision benefits, not equal-label information. The multihead control has the same GP/reference encoder and 58 additional parameters (3.9%); task outputs are independent learned moment heads. This is a parameter-sharing control, not a GP-free neural baseline. D removes the chemical kernel term; E permits signed response slopes with unchanged network size; F omits contrast covariance in both training and inference.',
           '', '## Selected-checkpoint results','', '|Task|Model|AP|Within-pair AUROC|Macro MAE|','|---|---|---|---|---|']
    for task in [1,2]:
        for model in MODELS:
            r=summary.loc['selected',task,model]
            lines.append('|'+str(task)+'|'+model+'|'+'|'.join(f'{r[m,"mean"]:.4f} ± {r[m,"std"]:.4f}' for m in METRICS)+'|')
    lines+=['','## Paired differences: Joint minus control','','|Task|Control|Metric|Difference|95% CI|Seeds better|','|---|---|---|---|---|---|']
    for r in ci.query('checkpoint=="selected"').itertuples():
        lines.append(f'|{r.task}|{r.control}|{r.metric}|{r.joint_minus_control:+.4f}|[{r.CI_low:+.4f}, {r.CI_high:+.4f}]|{r.seeds_joint_better}/5|')
    lines+=['','Intervals resample connected assay/document dependency components with one shared weight vector across all five seeds, both tasks, all models and checkpoints. Seeds are dependent repeats, not independent observations. AP is pooled within each seed then averaged across seeds; within-pair AUROC requires ≥10 queries and ≥3 per class; MAE is pair-macro then seed-macro. An unevaluable seed invalidates that metric draw instead of changing seed weighting. Intervals are conditional on the fitted models; no multiple-comparison correction is applied.',
            '', 'All seed and pair outcomes, including negative differences, are retained. The matched-label comparison and covariance ablation determine whether a shared posterior is helpful; this experiment cannot establish that it is necessary.']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstrap-draws',type=int,default=2000)
    main(p.parse_args().bootstrap_draws)
