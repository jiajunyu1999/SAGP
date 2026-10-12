"""Recompute a broad comparison from aligned per-query predictions.

Every listed architecture has all five seeds. Task 3 reference draws are
averaged within seed before SD; random-support and active-design comparisons
are kept separate. Strict within-pair AUROC always uses >=10 / >=3 per class.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'outputs/datasets/sar_transfer_20261007'
OUT=BASE/'paper/baseline_table_sources_final_20261009'
SEEDS=[17,23,41,71,101]
DEEP='models_deep_three_tasks_20261007'
ADD='models_additional_paper_baselines_20261008'
FIELD='models_known_response_field_20261007'
MONO='models_known_monotone_response_20261007'
JOINT='models_joint_response_field_20261007'
GP='models_meta_gp_transition_20261007'
COND='models_conditional_neural_gp_20261007'
SET='models_set_transformer_transition_20261007'
sources={}


def load(relative):
    path=DATA/relative
    sources[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    x=pd.read_csv(path)
    return x.rename(columns={'gap_B_target':'target','gap_B_mean':'regression'})


def scores(x,continuous=True):
    y=x.binary_loss.to_numpy();p=x.prediction.to_numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    if continuous:assert np.isfinite(x[['target','regression']].to_numpy()).all()
    strict=[];native=[]
    for _,g in x.groupby('pair_id',sort=False):
        n=len(g);pos=g.binary_loss.sum()
        if n>=5 and min(pos,n-pos)>=2:native.append(roc_auc_score(g.binary_loss,g.prediction))
        if n>=10 and min(pos,n-pos)>=3:strict.append(roc_auc_score(g.binary_loss,g.prediction))
    return dict(n=len(x),pairs=x.pair_id.nunique(),positive=int(y.sum()),
        AP=average_precision_score(y,p),AUROC=roc_auc_score(y,p),
        pair_macro_AUROC=np.mean(strict) if strict else np.nan,
        native_pair_AUROC=np.mean(native) if native else np.nan,
        strict_pairs=len(strict),native_pairs=len(native),
        macro_MAE=float(x.assign(error=abs(x.target-x.regression)).groupby('pair_id').error.mean().mean()) if continuous else np.nan)


def align(frame,canonical,task):
    key='sample_id' if task==2 else 'comparison_id'
    assert not frame[key].duplicated().any()
    assert set(frame[key])==set(canonical[key]),(task,'Different test cohorts')
    frame=frame.set_index(key).loc[canonical[key]].reset_index()
    for c in ['pair_id','binary_loss']:
        assert frame[c].equals(canonical[c]),(task,c)
    if 'target' in frame:
        assert np.allclose(frame.target,canonical.target,atol=1e-7),(task,'target mismatch')
    return frame


def known_models(task):
    controls=[(DEEP,'retain_gap_A','Constant risk / unchanged A gap')] if task==1 else [(DEEP,'pair_prior','Assay-pair prior')]
    controls += [(ADD,'linear','Logistic regression / ridge'),(ADD,'random_forest','Random forest'),(DEEP,'local_knn','Within-pair molecular kNN')]
    controls += [(DEEP,'contrast_boost_with_gap_regression','Contrast boosting')] if task==1 else [(DEEP,'xgboost','XGBoost')]
    if task==1:controls += [('models_threshold_curve_20261007','threshold_monotone','Threshold-monotone boosting')]
    controls += [(DEEP,'direct_neural','Direct MLP'),(DEEP,'reference_attention','Reference attention'),
        (DEEP,'ebayes_local','Neural local empirical Bayes'),
        (FIELD,'fixed_tanimoto_gp','Fixed Tanimoto GP'),(FIELD,'neural_mean_gp','Neural-mean GP'),
        (MONO,'global_monotone','Global monotone GP'),(MONO,'conditional_monotone','Conditional monotone GP')]
    if task==2:controls += [(MONO,'reference_gaussian_median','Reference Gaussian calibration')]
    controls += [(JOINT,'joint_monotone','Joint monotone GP (previous)')]
    controls += [('protected_decision_gp_20261009','detached_action','GP with global decision calibration'),('contextual_decision_gp_20261009','context_neural','Reference-conditioned Decision GP (ours)')]
    return controls


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];registry=[]
    for task in [1,2]:
        for directory,name,label in known_models(task):
            registry.append(dict(task=task,model=name,label=label,policy='known_pair'))
            for seed in SEEDS:
                canonical=load(f'{JOINT}/task{task}_joint_monotone_test_seed_{seed}.csv')
                if name=='threshold_monotone':
                    x=load(f'{directory}/predictions_{name}_test_seed_{seed}.csv')
                else:x=load(f'{directory}/task{task}_{name}_selected_test_seed_{seed}.csv') if name in ['detached_action','context_linear','context_neural'] else load(f'{directory}/task{task}_{name}_test_seed_{seed}.csv')
                x=align(x,canonical,task)
                rows.append(dict(task=task,model=name,label=label,policy='known_pair',seed=seed,draw=0,**scores(x,name!='threshold_monotone')))
    models=[('linear','Linear-kernel GP'),('tanimoto','Fixed Tanimoto GP'),('tanimoto_student','Student-t Tanimoto process'),
        ('deep','Deep-kernel GP'),('tanimoto_deep','Tanimoto + deep kernel GP'),('tanimoto_deep_metric','Learned molecular-metric GP'),
        ('conditional_mean_point','Conditional neural-mean GP'),('conditional_root_point','Root-supervised conditional GP (ours)')]
    for name,label in models:
        registry.append(dict(task=3,model=name,label=label,policy='contrast_design'))
        for seed in SEEDS:
            for draw in range(3):
                canonical=load(f'{COND}/predictions_conditional_root_point_contrast_design_k5_seed_{seed}_draw{draw}.csv')
                if name.startswith('conditional_'):x=load(f'{COND}/predictions_{name}_contrast_design_k5_seed_{seed}_draw{draw}.csv')
                else:
                    suffix=f'_draw{draw}' if draw else ''
                    x=load(f'{GP}/design_predictions_{name}_contrast_design_k5_seed_{seed}{suffix}.csv')
                x=align(x,canonical,3)
                rows.append(dict(task=3,model=name,label=label,policy='contrast_design',seed=seed,draw=draw,**scores(x)))
    for name,label in [('fixed_tanimoto','Fixed Tanimoto GP'),('set_transformer_pw0.75','Set Transformer'),('conditional_root_point','Root-supervised conditional GP (ours)')]:
        registry.append(dict(task=3,model=name,label=label,policy='random'))
        for seed in SEEDS:
            for draw in range(3):
                canonical=load(f'{SET}/predictions_fixed_tanimoto_test_seed_{seed}_draw{draw}.csv')
                x=align(load(f'{SET}/predictions_{name}_test_seed_{seed}_draw{draw}.csv'),canonical,3)
                rows.append(dict(task=3,model=name,label=label,policy='random',seed=seed,draw=draw,**scores(x)))
    allrows=pd.DataFrame(rows)
    allrows.to_csv(OUT/'metrics_by_seed_and_draw.csv',index=False)
    keys=['task','model','label','policy']
    metrics=['AP','AUROC','pair_macro_AUROC','native_pair_AUROC','macro_MAE']
    perseed=allrows.groupby(keys+['seed'],sort=False)[metrics].mean().reset_index()
    perseed.to_csv(OUT/'metrics_by_seed.csv',index=False)
    summary=perseed.groupby(keys,sort=False)[metrics].agg(['mean','std'])
    summary.to_csv(OUT/'comparison_mean_sd.csv')
    pd.DataFrame(registry).to_csv(OUT/'model_registry.csv',index=False)
    tables=[]
    for task in [1,2,3]:
        title={1:'Known-pair SAR transfer',2:'Molecular sensitivity within a known assay pair',3:'Five-reference transfer to held-out assay/source contexts'}[task]
        tables += [r'\begin{table}[htbp]',r'\centering',rf'\caption{{{title}. Results are mean $\pm$ SD across five fixed scaffold splits.}}',rf'\label{{tab:baselines-task{task}}}',r'\small',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & AP $\uparrow$ & AUROC $\uparrow$ & Pair AUROC $\uparrow$ & Macro MAE $\downarrow$ \\',r'\midrule']
        policies=['known_pair'] if task!=3 else ['contrast_design','random']
        for policy in policies:
            selected=[r for r in registry if r['task']==task and r['policy']==policy]
            if task==3:
                text='Contrast-variance reference selection' if policy=='contrast_design' else 'Identical random references (Transformer control)'
                tables.append(r'\multicolumn{5}{l}{\textit{'+text+r'}} \\')
            block=summary.loc[(task,slice(None),slice(None),policy),:]
            for r in selected:
                stat=summary.loc[(task,r['model'],r['label'],policy)]
                cells=[]
                for metric in ['AP','AUROC','pair_macro_AUROC','macro_MAE']:
                    mean,sd=stat[(metric,'mean')],stat[(metric,'std')]
                    best=block[(metric,'mean')].min() if metric=='macro_MAE' else block[(metric,'mean')].max()
                    cell='--' if pd.isna(mean) else rf'{mean:.3f} $\pm$ {sd:.3f}'
                    if pd.notna(mean) and np.isclose(mean,best,atol=1e-10):cell=r'\textbf{'+cell+'}'
                    cells.append(cell)
                if '(ours)' in r['label']:tables.append(r'\midrule')
                tables.append(r['label']+' & '+' & '.join(cells)+r' \\')
            if task==3 and policy=='contrast_design':tables.append(r'\midrule')
        tables += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',r'\begin{minipage}{0.98\textwidth}\footnotesize']
        note='Pair AUROC uses the same strict rule for every row: at least ten queries and three per class. Bold marks the largest mean for ranking metrics or smallest mean error; it does not indicate statistical significance. '
        if task in [1,2]:
            note+='All methods use identical test IDs and outcomes within each seed. Linear and forest classification/regression estimators address separate endpoints; their outputs are never fused. '
            if task==1:note+='The threshold classifier has no continuous output; its companion gap regressor is listed under contrast boosting. '
        else:
            note+='Three reference draws are averaged within each seed before computing SD. Both blocks use five references. Active selection uses the same policy and candidate pools, but selected reference identities may differ between models. The random block uses identical reference identities across its three methods. Native five-query/two-per-class AUROC is exported separately. '
        note+='Training objectives and available reference representations differ; the tables compare task-level prediction methods rather than isolating architecture alone.'
        tables += [note,r'\end{minipage}',r'\end{table}','']
    (BASE/'paper/baseline_comparison_tables.tex').write_text('\n'.join(tables))
    (OUT/'manifest.json').write_text(json.dumps(dict(source_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seeds=SEEDS,
        models_per_task={str(t):sum(r['task']==t and r['policy']!='random' for r in registry) for t in [1,2,3]},
        task3_draws='Average three draws within each seed; SD over five seeds',alignment='Exact per-seed test IDs, pair IDs, class labels and continuous targets checked',
        strict_pair_rule='at least 10 queries, at least 3 per class',selection='Selected comparison set; additional evaluated variants remain archived',excluded_from_presentation=['calibrated_attention'],exclusion_reason='User-requested presentation scope; original experimental records preserved'),indent=2))
    print(summary.round(4).to_string())


if __name__=='__main__':main()
