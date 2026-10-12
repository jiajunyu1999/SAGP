"""Twelve observed-data panels: model coverage, pair heterogeneity and utility.

No fitted-model averaging: averaging here summarizes repeated evaluations.
The five splits are dependent repeats, and their SD is descriptive variability.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from make_final_baseline_tables_20261009 import (
    BASE,DATA,SEEDS,DEEP,ADD,FIELD,MONO,JOINT,GP,COND,align,known_models,
)

OUT=BASE/'outputs/figures'
AUDIT=OUT/'figure3_unfiltered_main_20261012'
AUDIT.mkdir(exist_ok=True)
from publication_palette_20261009 import BLUE,RED,PEACH,YELLOW,INK,GRAY,REFERENCE_PALETTE
TEAL,ORANGE=PEACH,RED
PALETTE=[RED,BLUE,PEACH]
RNG=np.random.default_rng(20261008)
SOURCES={}
CACHE={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.1,'axes.labelsize':10.6,
    'xtick.labelsize':9.6,'ytick.labelsize':9.6,'axes.linewidth':.65,
    'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,
    'svg.fonttype':'none','axes.unicode_minus':False})

SHORT={'retain_gap_A':'Constant prior','pair_prior':'Pair prior','linear':'Linear',
    'random_forest':'Random forest','local_knn':'Molecular kNN','xgboost':'XGBoost',
    'contrast_boost_with_gap_regression':'ContrastBoost','threshold_monotone':'ThresholdBoost',
    'direct_neural':'MLP','reference_attention':'Ref. attention','ebayes_local':'Local Bayes',
    'fixed_tanimoto_gp':'Tanimoto GP','neural_mean_gp':'Neural-mean GP',
    'global_monotone':'GlobalMono GP','conditional_monotone':'CondMono GP',
    'reference_gaussian_median':'Gaussian calib.','joint_monotone':'Joint GP (previous)','detached_action':'Global decision GP','context_neural':'SAGP',
    'tanimoto':'Tanimoto GP','tanimoto_student':'Student-t GP','deep':'Deep GP',
    'tanimoto_deep':'Tanimoto + deep','tanimoto_deep_metric':'Metric GP',
    'conditional_mean_point':'Neural-mean GP','conditional_root_point':'Root GP (ours)',
    'fixed_tanimoto':'Tanimoto GP','set_transformer_pw0.75':'Set Transformer'}
RANK_CONTROLS={1:['random_forest','local_knn','direct_neural'],
    2:['random_forest','local_knn','direct_neural'],3:['linear','tanimoto','tanimoto_student']}
ERROR_CONTROLS={1:'contrast_boost_with_gap_regression',2:'local_knn',3:'tanimoto'}
UTILITY_CONTROLS={1:['context_neural','random_forest','local_knn'],
    2:['context_neural','random_forest','local_knn'],
    3:['conditional_root_point','tanimoto','tanimoto_student']}


def read(path):
    path=Path(path)
    SOURCES[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path).rename(columns={'gap_B_target':'target','gap_B_mean':'regression'})


def prediction(task,model,seed,draw=0):
    key=(task,model,seed,draw)
    if key in CACHE:return CACHE[key]
    if task<3:
        directory=next(d for d,n,_ in known_models(task) if n==model)
        filename=f'predictions_{model}_test_seed_{seed}.csv' if model=='threshold_monotone' else (f'task{task}_{model}_selected_test_seed_{seed}.csv' if model in ['detached_action','context_linear','context_neural'] else f'task{task}_{model}_test_seed_{seed}.csv')
    elif model.startswith('conditional_'):
        directory=COND;filename=f'predictions_{model}_contrast_design_k5_seed_{seed}_draw{draw}.csv'
    else:
        directory=GP;suffix=f'_draw{draw}' if draw else ''
        filename=f'design_predictions_{model}_contrast_design_k5_seed_{seed}{suffix}.csv'
    x=read(DATA/directory/filename)
    assert np.isfinite(x.prediction).all() and x.prediction.between(0,1).all()
    CACHE[key]=x
    return x


def pair_records(task,control):
    rows=[];ours='context_neural' if task<3 else 'conditional_root_point'
    for seed in SEEDS:
        for draw in range(3 if task==3 else 1):
            a=prediction(task,ours,seed,draw)
            b=align(prediction(task,control,seed,draw),a,task)
            for pair,ix in a.groupby('pair_id').indices.items():
                aa=a.iloc[ix];bb=b.iloc[ix];y=aa.binary_loss.to_numpy()
                eligible=len(y)>=10 and min(y.sum(),len(y)-y.sum())>=3
                gain=roc_auc_score(y,aa.prediction)-roc_auc_score(y,bb.prediction) if eligible else np.nan
                ae=abs(aa.target-aa.regression).mean()
                be=abs(bb.target-bb.regression).mean() if 'regression' in bb else np.nan
                rows.append(dict(task=task,baseline=control,seed=seed,draw=draw,pair_id=pair,
                    n_queries=len(y),positives=int(y.sum()),auc_gain=gain,ours_MAE=ae,baseline_MAE=be))
    return pd.DataFrame(rows)


def clean(ax,xgrid=False):
    for s in ['top','right']:ax.spines[s].set_visible(False)
    ax.tick_params(width=.65,length=3,pad=3)
    ax.set_axisbelow(True)
    if xgrid:ax.grid(axis='x',color=GRAY,lw=.5,ls='--',alpha=.65,zorder=0)


def letter(ax,label):
    ax.text(-.08,1.065,label,transform=ax.transAxes,fontsize=13,fontweight='bold',color=INK)


def macro_recall(frame,fractions):
    curves=[]
    for _,g in frame.groupby('pair_id'):
        n=len(g);positives=int(g.binary_loss.sum())
        if n<10 or min(positives,n-positives)<3:continue
        score=g.prediction.to_numpy();y=g.binary_loss.to_numpy()
        order=np.argsort(-score,kind='stable');score=score[order];y=y[order]
        starts=np.r_[0,np.flatnonzero(np.diff(score)!=0)+1]
        sizes=np.diff(np.r_[starts,n]);tp=np.add.reduceat(y,starts)
        # Linear interpolation within tied-score groups gives expected recall
        # for a uniform tie break and an exact fractional per-pair budget.
        recall=np.interp(fractions*n,np.r_[0,np.cumsum(sizes)],np.r_[0,np.cumsum(tp)])/positives
        curves.append(recall)
    assert curves
    return np.mean(curves,axis=0),len(curves)


def main():
    metric_path=BASE/'paper/baseline_table_sources_final_20261009/metrics_by_seed.csv'
    registry_path=BASE/'paper/baseline_table_sources_final_20261009/model_registry.csv'
    metrics=read(metric_path);registry=read(registry_path)
    excluded={'reference_gaussian_median','joint_monotone','detached_action','fixed_tanimoto_gp','neural_mean_gp','global_monotone','conditional_monotone'}
    metrics=metrics[~metrics.model.isin(excluded)].copy();registry=registry[~registry.model.isin(excluded)].copy()
    assert not registry.model.eq('calibrated_attention').any()
    fig,axes=plt.subplots(4,2,figsize=(9.8,12.3),
        gridspec_kw={'height_ratios':[4.2,1.9,2.05,2.05]})
    fig.subplots_adjust(left=.13,right=.985,bottom=.055,top=.90,hspace=.68,wspace=.88)
    summaries={};pair_runs=[];pair_dots=[];error_dots=[];utility=[]
    for col,task in enumerate([1,2]):
        ax=axes[0,col];clean(ax,True);ax.spines['left'].set_visible(False)
        chosen=registry.query('task==@task');names=chosen.model.tolist()
        for row,(_,setting) in enumerate(chosen.iterrows()):
            model=setting.model;policy=setting.policy
            if task==3 and row==8:ax.axhline(7.5,color=GRAY,lw=.8,ls='--',zorder=1)
            ours=model in ['context_neural','conditional_root_point']
            if ours:ax.axhspan(row-.44,row+.44,color=YELLOW,zorder=0)
            values=metrics.query('task==@task and model==@model and policy==@policy')
            assert values.seed.nunique()==5 and len(values)==5
            for offset,metric,color in [(-.17,'AP',BLUE),(.17,'pair_macro_AUROC',TEAL)]:
                v=values[metric].to_numpy();y=row+offset
                ax.scatter(v,y+RNG.uniform(-.065,.065,5),s=11,color=color,alpha=.55,linewidths=0,zorder=3)
                mean,sd=v.mean(),v.std(ddof=1)
                ax.plot([mean-sd,mean+sd],[y,y],color=color,lw=1.,alpha=.8,zorder=2)
                ax.scatter([mean],[y],s=24,facecolor='white',edgecolor=color,linewidth=1.05,zorder=4)
        ax.set_yticks(range(len(names)),[SHORT[r.model]+(' [R]' if r.policy=='random' else '') for _,r in chosen.iterrows()],fontsize=9.5)
        for label,name in zip(ax.get_yticklabels(),names):
            if name in ['context_neural','conditional_root_point']:label.set_fontweight('bold')
        ax.tick_params(axis='y',length=0)
        ax.set_ylim(len(names)-.4,-.6);ax.set_xlim(0,1.02);ax.set_xticks([0,.5,1.])
        ax.set_xlabel('Test ranking score')
        letter(ax,chr(ord('a')+col))
        bb=ax.get_position()
        title=['Task 1 | SAR transfer','Task 2 | Molecular sensitivity'][col]
        fig.text((bb.x0+bb.x1)/2,.974,title,ha='center',fontsize=10.5,fontweight='bold',color=INK)
        summaries[str(task)]={'model_policy_settings':len(names),'model_seed_points':len(names)*10}
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='AP'),
                        Line2D([],[],marker='o',ls='',color=TEAL,label='Within-pair AUROC'),
                        Line2D([],[],marker='o',mfc='white',mec=INK,color=INK,lw=.9,label='Mean ± SD; small dots: five seeds')],
               loc='upper center',bbox_to_anchor=(.55,.961),ncol=3,frameon=False,fontsize=9.6,
               handlelength=1.5,columnspacing=1.5)
    max_gain=0
    for col,task in enumerate([1,2]):
        ax=axes[1,col];clean(ax,True);ax.spines['left'].set_visible(False)
        labels=[]
        for row,control in enumerate(RANK_CONTROLS[task]):
            run=pair_records(task,control);pair_runs.append(run)
            dots=run.groupby(['task','baseline','pair_id'],as_index=False).auc_gain.mean().dropna()
            pair_dots.append(dots);v=dots.auc_gain.to_numpy();max_gain=max(max_gain,float(abs(v).max()))
            ax.scatter(v,row+RNG.uniform(-.18,.18,len(v)),s=12,alpha=.60,color=PALETTE[row],linewidths=0,rasterized=True,zorder=3)
            q1,med,q3=np.quantile(v,[.25,.5,.75])
            ax.plot([q1,q3],[row+.29,row+.29],color=PALETTE[row],lw=1.6,zorder=4)
            ax.scatter([med],[row+.29],s=23,facecolor='white',edgecolor=PALETTE[row],linewidth=1,zorder=5)
            labels.append(f'{SHORT[control]}\n{len(v)} pairs; {(v>0).mean():.0%} gain')
        ax.set_yticks(range(3),labels,fontsize=9.1);ax.tick_params(axis='y',length=0)
        ax.set_ylim(2.65,-.55);ax.axvline(0,color='#8D969E',lw=.75,ls='--',zorder=1)
        ax.set_xlabel('Our − baseline pair AUROC')
        letter(ax,chr(ord('c')+col))
    lim=max(.1,max_gain*1.08)
    for ax in axes[1]:ax.set_xlim(-lim,lim);ax.set_xticks([-round(lim*.7,2),0,round(lim*.7,2)])
    for col,task in enumerate([1,2]):
        control=ERROR_CONTROLS[task];run=pair_records(task,control)
        errors=run.groupby(['task','baseline','pair_id'],as_index=False).agg(
            ours_MAE=('ours_MAE','mean'),baseline_MAE=('baseline_MAE','mean'),runs=('seed','size'))
        error_dots.append(errors);ax=axes[2,col];clean(ax)
        x=errors.baseline_MAE.to_numpy();y=errors.ours_MAE.to_numpy()
        ceiling=max(x.max(),y.max())*1.14
        ax.plot([0,ceiling],[0,ceiling],ls='--',lw=.8,color='#8D969E',zorder=1)
        ax.scatter(x,y,s=13,color=BLUE,alpha=.60,linewidths=0,zorder=3,rasterized=True)
        ax.set(xlim=(0,ceiling),ylim=(0,ceiling))
        ax.set_xscale('symlog',linthresh=.08,linscale=.8)
        ax.set_yscale('symlog',linthresh=.08,linscale=.8)
        ticks=[v for v in [0,.1,.3,1,3] if v<=ceiling]
        ax.set_xticks(ticks,[str(v) for v in ticks]);ax.set_yticks(ticks,[str(v) for v in ticks])
        ax.set_xlabel(f'{SHORT[control]} MAE (pIC50)',fontsize=9.1)
        ax.set_ylabel('Our MAE (pIC50)',fontsize=9.1)
        ax.text(.04,.96,f'{len(errors)} pairs\n{(y<x).mean():.0%} lower error',transform=ax.transAxes,
                ha='left',va='top',fontsize=9,color=INK)
        letter(ax,chr(ord('e')+col))
        summaries[str(task)]['error_pairs']=len(errors)
    fractions=np.linspace(0,1,21)
    for col,task in enumerate([1,2]):
        ax=axes[3,col];clean(ax);ax.plot([0,1],[0,1],ls='--',lw=.85,color='#A3ABB2',zorder=1)
        for j,model in enumerate(UTILITY_CONTROLS[task]):
            seed_curves=[]
            for seed in SEEDS:
                draws=[]
                for draw in range(3 if task==3 else 1):
                    curve,n=macro_recall(prediction(task,model,seed,draw),fractions);draws.append(curve)
                    for budget,recall in zip(fractions,curve):utility.append(dict(task=task,model=model,seed=seed,draw=draw,budget=budget,recall=recall,eligible_pairs=n))
                curve=np.mean(draws,axis=0);seed_curves.append(curve)
                ax.plot(fractions,curve,color=PALETTE[j],alpha=.16,lw=.7,zorder=2)
                indices=[2,4,8,12,16]
                ax.scatter(fractions[indices],curve[indices],s=8,color=PALETTE[j],alpha=.35,linewidths=0,zorder=3)
            values=np.stack(seed_curves);mean=values.mean(0)
            ax.plot(fractions,mean,color=PALETTE[j],lw=1.65,label=SHORT[model],zorder=4)
            if j==0:
                at10=float(mean[2]);summaries[str(task)]['recall_at_10_percent']=at10
                summaries[str(task)]['enrichment_at_10_percent']=at10/.1
                ax.text(.035,1.045,f'Top 10%: {at10:.0%} recalled\n{at10/.1:.1f}× random',
                        transform=ax.transAxes,va='bottom',ha='left',fontsize=9,color=INK)
        ax.set_xlim(0,1);ax.set_ylim(0,1.03);ax.set_xticks([0,.5,1],['0','50','100'])
        ax.set_yticks([0,.5,1],['0','50','100'])
        ax.set_xlabel('Queries selected per pair (%)',fontsize=9.2)
        ax.set_ylabel('Positive queries recalled (%)',fontsize=9.2)
        ax.legend(frameon=False,fontsize=9.0,loc='lower right',handlelength=1.3,labelspacing=.3)
        letter(ax,chr(ord('g')+col))
    metrics.to_csv(AUDIT/'baseline_seed_points.csv',index=False)
    pd.concat(pair_runs,ignore_index=True).to_csv(AUDIT/'pair_gain_runs.csv',index=False)
    pd.concat(pair_dots,ignore_index=True).to_csv(AUDIT/'unique_pair_gain_points.csv',index=False)
    pd.concat(error_dots,ignore_index=True).to_csv(AUDIT/'unique_pair_error_points.csv',index=False)
    pd.DataFrame(utility).to_csv(AUDIT/'within_pair_query_recall.csv',index=False)
    for extension in ['pdf','png','svg']:
        fig.savefig(OUT/f'figure3_model_gains.{extension}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    manifest=dict(palette=REFERENCE_PALETTE,palette_source="paper/image.png",
        palette_script_sha256=hashlib.sha256((BASE/'analysis/publication_palette_20261009.py').read_bytes()).hexdigest(),panels=8,tasks=summaries,source_sha256=SOURCES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jitter_seed=20261008,
        ranking_rule='>=10 queries, >=3 per class; all positive and negative gains retained',
        pair_dots='One unique assay pair; mean across available test runs; unweighted across unique pairs',
        error_dots='All test assay pairs, including pairs not eligible for binary AUROC; average MAE across available runs',
        utility='Within-pair fractional query-selection budget; uniform expected tie break; macro recall over strict-eligible pairs; draw-average then equal seed average',
        statistics='Seed SD is descriptive, not an independent-replicate CI; median/IQR summarize unique pair gains',
        exclusion='Calibrated reference attention excluded from presentation as requested; original records retained')
    (AUDIT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (AUDIT/'README.txt').write_text('Two-task figure: a/b scores; c/d pair gains; e/f pair errors; g/h query recall. All pair gain signs retained. Five scaffold splits are dependent repeats.\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
