"""Validation-only model selection and shared-source bootstrap for three tasks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from meta_gp_sar_20261007 import context_components,read,RESULTS as GP,SEEDS
from deep_three_tasks_20261007 import RESULTS

NEURAL=['direct_neural','reference_attention','anchored_direct','anchored_attention','projected_attention']
GP_NEURAL=['deep','tanimoto_deep','tanimoto_deep_metric']


def bootstrap(records,roots,reps=1000):
    packed=[];contexts=set()
    for a,b in records:
        idcol='comparison_id' if 'comparison_id' in a else 'sample_id'
        assert a[idcol].equals(b[idcol]);assert a.binary_loss.equals(b.binary_loss)
        c=a.pair_id.map(roots).to_numpy();assert not pd.isna(c).any();contexts.update(c)
        y=a.binary_loss.to_numpy();pair=[]
        for name,ix in a.groupby('pair_id').indices.items():
            yy=y[ix]
            if len(ix)>=10 and yy.sum()>=3 and (1-yy).sum()>=3:
                pair.append((roots[name],roc_auc_score(yy,a.prediction.to_numpy()[ix])-roc_auc_score(yy,b.prediction.to_numpy()[ix])))
        ed=pd.DataFrame({'pair_id':a.pair_id,'error':abs(a.target-a.regression)-abs(b.target-b.regression)}).groupby('pair_id').error.mean()
        packed.append((y,a.prediction.to_numpy(),b.prediction.to_numpy(),c,pair,ed))
    contexts=sorted(contexts);lookup={c:i for i,c in enumerate(contexts)}
    prepared=[]
    for y,pa,pb,c,pair,err in packed:
        prepared.append((y,pa,pb,np.array([lookup[v] for v in c]),
            np.array([lookup[v[0]] for v in pair]),np.array([v[1] for v in pair]),
            np.array([lookup[roots[p]] for p in err.index]),err.to_numpy()))
    rng=np.random.default_rng(20261007);output=[]
    for run in range(reps+1):
        w=np.ones(len(contexts)) if run==0 else rng.multinomial(len(contexts),np.full(len(contexts),1/len(contexts)))
        vals=[]
        for y,pa,pb,c,pc,pv,ec,ev in prepared:
            sw=w[c]
            if (sw*y).sum()==0 or (sw*(1-y)).sum()==0:continue
            ap=average_precision_score(y,pa,sample_weight=sw)-average_precision_score(y,pb,sample_weight=sw)
            auc=np.average(pv,weights=w[pc]) if len(pv) and w[pc].sum()>0 else np.nan
            mae=np.average(ev,weights=w[ec]) if w[ec].sum()>0 else np.nan
            vals.append([ap,auc,mae])
        output.append(np.nanmean(vals,axis=0))
    out=np.array(output)
    return {'source_components':len(contexts),'replicates':reps,
        'difference':dict(zip(['AP','pair_macro_AUROC','macro_MAE'],out[0].tolist())),
        'bootstrap_95_CI':dict(zip(['AP','pair_macro_AUROC','macro_MAE'],np.nanquantile(out[1:],[.025,.975],axis=0).T.tolist())),
        'note':'Shared assay/document-component weights across seeds and reference draws; conditional on fitted models and validation selections. Exploratory.'}


def main():
    roots=context_components(read('molecule_assay_observations.parquet'))
    metrics=pd.read_csv(RESULTS/'metrics.csv');test=metrics.query("partition=='test'")
    agg=test.groupby(['task','model']).agg(**{k:(k,'mean') for k in ['AP','AUROC','pair_macro_AUROC','macro_MAE']},
        AP_SD=('AP','std'),AUROC_SD=('AUROC','std'),runs=('AP','size'))
    agg.to_csv(RESULTS/'summary_by_model.csv')
    rows=[];selection=[];comparisons=[]
    for task in [1,2]:
        records_by_baseline={name:[] for name in (['contrast_boost_with_gap_regression'] if task==1 else ['local_knn','xgboost','pair_prior'])}
        for seed in SEEDS:
            va=metrics[(metrics.task==task)&(metrics.seed==seed)&(metrics.partition=='validation')&metrics.model.isin(NEURAL)]
            key='AP' if task==1 else 'pair_macro_AUROC'
            chosen=va.sort_values(key,ascending=False).iloc[0].model
            selection.append({'task':task,'seed':seed,'model':chosen,'validation_metric':key,'validation_score':float(va.set_index('model').loc[chosen,key])})
            r=test[(test.task==task)&(test.seed==seed)&(test.model==chosen)].iloc[0].to_dict();r['model']='validation_selected_neural';rows.append(r)
            a=pd.read_csv(RESULTS/f'task{task}_{chosen}_test_seed_{seed}.csv')
            for baseline in records_by_baseline:
                b=pd.read_csv(RESULTS/f'task{task}_{baseline}_test_seed_{seed}.csv')
                records_by_baseline[baseline].append((a,b))
        for baseline,records in records_by_baseline.items():
            result=bootstrap(records,roots);result.update({'task':task,'neural':'validation_selected_neural','baseline':baseline});comparisons.append(result)
    gp_metrics=pd.read_csv(GP/'metrics.csv')
    design=[]
    for draw in range(3):
        x=pd.read_csv(GP/('design_metrics.csv' if draw==0 else f'design_metrics_draw{draw}.csv'));x['draw']=draw;design.append(x)
    design=pd.concat(design,ignore_index=True)
    dsum=design.groupby(['model','policy','support_k']).agg(**{k:(k,'mean') for k in ['AP','AUROC','macro_MAE','macro_NLL','coverage_90']},runs=('AP','size'))
    dsum.to_csv(GP/'design_summary.csv')
    records=[]
    for seed in SEEDS:
        va=gp_metrics[(gp_metrics.seed==seed)&(gp_metrics.partition=='validation')&(gp_metrics.support_k==5)&gp_metrics.model.isin(GP_NEURAL)]
        chosen=va.sort_values('macro_NLL').iloc[0].model
        selection.append({'task':3,'seed':seed,'model':chosen,'validation_metric':'macro_NLL','validation_score':float(va.set_index('model').loc[chosen,'macro_NLL'])})
        r=design[(design.seed==seed)&(design.model==chosen)&(design.policy=='contrast_design')&(design.support_k==5)].mean(numeric_only=True).to_dict()
        r.update({'task':3,'seed':seed,'model':'validation_selected_neural','partition':'test'});rows.append(r)
        for draw in range(3):
            suffix=f'_draw{draw}' if draw else ''
            a=pd.read_csv(GP/f'design_predictions_{chosen}_contrast_design_k5_seed_{seed}{suffix}.csv')
            b=pd.read_csv(GP/f'design_predictions_tanimoto_contrast_design_k5_seed_{seed}{suffix}.csv')
            for f in [a,b]:f['target']=f.gap_B_target;f['regression']=f.gap_B_mean
            records.append((a,b))
    result=bootstrap(records,roots);result.update({'task':3,'neural':'validation_selected_neural','baseline':'tanimoto_contrast_design_k5'});comparisons.append(result)
    selected=pd.DataFrame(rows);selected.to_csv(RESULTS/'selected_neural_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(RESULTS/'validation_selection.csv',index=False)
    sm=selected.groupby('task').agg(**{k:(k,'mean') for k in ['AP','AUROC','pair_macro_AUROC','macro_MAE']},AP_SD=('AP','std'),AUROC_SD=('AUROC','std'))
    sm.to_csv(RESULTS/'selected_neural_summary.csv')
    (RESULTS/'source_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    print('ALL MODELS\n',agg.round(4).to_string(),flush=True)
    print('VALIDATION SELECTED\n',sm.round(4).to_string(),flush=True)
    print('SELECTION\n',pd.DataFrame(selection).to_string(index=False),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':main()
