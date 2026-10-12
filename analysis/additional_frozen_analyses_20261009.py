"""Frozen post-hoc diagnostics: no fitting, model selection or test threshold tuning."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from make_final_baseline_tables_20261009 import known_models,DATA,BASE,SEEDS,scores
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_palette_20261009 import RED,PEACH,BLUE,LIGHT_BLUE,INK,GRAY
OUT=DATA/'additional_frozen_analyses_20261009'
MODELS=['context_neural','joint_monotone','detached_action','fixed_tanimoto_gp','local_knn','random_forest']
COLORS={'context_neural':RED,'joint_monotone':PEACH,'detached_action':BLUE,'fixed_tanimoto_gp':LIGHT_BLUE,'local_knn':'#9BB5AE','random_forest':'#A3A3A3'}
LABELS={'context_neural':'Decision GP','joint_monotone':'Joint GP','detached_action':'Global readout','fixed_tanimoto_gp':'Tanimoto GP','local_knn':'Molecular kNN','random_forest':'Random forest'}
SIM_EDGES=[0,.3,.5,.7,.8,1.00001];SIM_LABELS=['0–0.3','0.3–0.5','0.5–0.7','0.7–0.8','0.8–1.0']
SOURCES={}
def load(p):
 SOURCES[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def calibration(y,p):
 p=np.clip(np.asarray(p,float),1e-7,1-1e-7);y=np.asarray(y)
 b=np.minimum((p*10).astype(int),9);ece=sum(np.mean(b==i)*abs(y[b==i].mean()-p[b==i].mean()) for i in range(10) if np.any(b==i))
 return {'Brier':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),'ECE10':ece}
def evaluate(x):
 if len(x)==0:return dict(n=0,pairs=0,AP=np.nan,AUROC=np.nan,pair_macro_AUROC=np.nan,macro_MAE=np.nan)
 result=dict(n=len(x),pairs=x.pair_id.nunique(),AP=np.nan,AUROC=np.nan,pair_macro_AUROC=np.nan,macro_MAE=np.nan)
 if x.binary_loss.nunique()==2:
  result.update(AP=average_precision_score(x.binary_loss,x.prediction),AUROC=roc_auc_score(x.binary_loss,x.prediction))
 auc=[]
 for _,g in x.groupby('pair_id',sort=False):
  pos=g.binary_loss.sum()
  if len(g)>=10 and min(pos,len(g)-pos)>=3:auc.append(roc_auc_score(g.binary_loss,g.prediction))
 result['pair_macro_AUROC']=np.mean(auc) if auc else np.nan;result['eligible_pairs']=len(auc)
 if 'regression' in x:result['macro_MAE']=x.assign(error=abs(x.target-x.regression)).groupby('pair_id').error.mean().mean()
 return result

def main():
 OUT.mkdir(exist_ok=True);threadpool_limits(limits=4)
 obs=pd.read_parquet(DATA/'molecule_assay_observations.parquet');anchor=pd.read_parquet(DATA/'anchor_transfer_all.parquet')
 npz=np.load(DATA/'protected_decision_gp_20261009/structure_features.npz');mapping={m:i for i,m in enumerate(npz['ids'])};bits=npz['bits'].astype(np.float32);sizes=bits.sum(1)
 meta2=obs.set_index('sample_id');meta1=anchor.set_index('comparison_id')
 rows=[];cal=[];reliability=[];risk=[];tail=[];threshold=[];enriched=[]
 for seed in SEEDS:
  ref=load(DATA/'controlled_joint_ablation_20261009'/f'reference_manifest_seed_{seed}.csv');groups={p:g for p,g in ref.groupby('pair_id',sort=False)}
  # Training-only reference pool defines target frequency; no test labels used.
  train_targets=obs.set_index('sample_id').loc[ref.sample_id,'target_chembl_id'];top=set(train_targets.value_counts().head(10).index)
  for task in [1,2]:
   key='comparison_id' if task==1 else 'sample_id';a=load(DATA/'contextual_decision_gp_20261009'/f'task{task}_context_neural_selected_test_seed_{seed}.csv');features=[]
   for pair,g in a.groupby('pair_id',sort=False):
    r=groups[pair];ri=np.array([mapping[i] for i in r.chemical_identity]);ids=g[key].to_numpy()
    if task==1:
     f=meta1.loc[ids];q1=np.array([mapping[i] for i in f.identity_1]);q2=np.array([mapping[i] for i in f.identity_2]);qi=np.unique(np.concatenate([q1,q2]))
    else:
     f=meta2.loc[ids];q1=np.array([mapping[i] for i in f.chemical_identity]);qi=np.unique(q1)
    intersection=bits[qi]@bits[ri].T;t=intersection/np.maximum(sizes[qi,None]+sizes[ri][None,:]-intersection,1);mx=dict(zip(qi,t.max(1)))
    similarity=np.array([mx[i] for i in q1])
    if task==1:similarity=np.minimum(similarity,[mx[i] for i in q2])
    for identity,sim in zip(ids,similarity):features.append({key:identity,'max_reference_tanimoto':sim,'n_references':len(r)})
   features=pd.DataFrame(features);a=a.merge(features,on=key,validate='one_to_one');a['seed']=seed;a['task']=task;enriched.append(a)
   for folder,model,_ in known_models(task):
    if model not in MODELS:continue
    if model=='context_neural':x=a.copy()
    else:
     filename=f'task{task}_{model}_selected_test_seed_{seed}.csv' if model=='detached_action' else f'task{task}_{model}_test_seed_{seed}.csv'
     x=load(DATA/folder/filename).rename(columns={'gap_B_target':'target','gap_B_mean':'regression'}).set_index(key).loc[a[key]].reset_index();assert np.array_equal(x.binary_loss,a.binary_loss)
     x=x.merge(features,on=key,validate='one_to_one')
    x['sim_bin']=pd.cut(x.max_reference_tanimoto,SIM_EDGES,labels=SIM_LABELS,include_lowest=True,right=False)
    for label in SIM_LABELS:
     g=x[x.sim_bin.eq(label)];rows.append(dict(task=task,seed=seed,model=model,sim_bin=label,**evaluate(g)))
    cal.append(dict(task=task,seed=seed,model=model,**calibration(x.binary_loss,x.prediction)))
    for i in range(10):
     g=x[np.minimum((np.clip(x.prediction,0,1)*10).astype(int),9)==i]
     reliability.append(dict(task=task,seed=seed,model=model,bin=i,n=len(g),mean_probability=g.prediction.mean(),observed_rate=g.binary_loss.mean()))
    for subset in ['all','exclude_top10','top10']:
     g=x if subset=='all' else x[x.target_chembl_id.isin(top).eq(subset=='top10')]
     tail.append(dict(task=task,seed=seed,model=model,subset=subset,targets=g.target_chembl_id.nunique(),**evaluate(g)))
    if model=='context_neural':
     cal.append(dict(task=task,seed=seed,model='uncalibrated_gaussian',**calibration(x.binary_loss,x.gaussian_probability)))
     for ordering in ['decision_entropy','posterior_variance','random']:
      p=np.clip(x.prediction.to_numpy(),1e-7,1-1e-7)
      score=-(p*np.log(p)+(1-p)*np.log1p(-p)) if ordering=='decision_entropy' else (x.variance.to_numpy() if ordering=='posterior_variance' else np.random.default_rng(seed+task).random(len(x)))
      order=np.argsort(score,kind='stable')
      for coverage in [.2,.4,.6,.8,1.]:
       g=x.iloc[order[:max(1,int(np.ceil(len(x)*coverage)))]]
       risk.append(dict(task=task,seed=seed,ordering=ordering,coverage=coverage,n=len(g),positive_rate=g.binary_loss.mean(),positive=int(g.binary_loss.sum()),classification_error=np.mean((g.prediction>=.5).astype(int)!=g.binary_loss),micro_MAE=np.mean(abs(g.target-g.regression)),pairs=g.pair_id.nunique()))
    if task==2 and 'regression' in x:
     for cut in [.3,.5,1.,1.5]:
      y=(x.target>=cut).astype(int);threshold.append(dict(seed=seed,model=model,threshold=cut,n=len(x),positive=int(y.sum()),AP=average_precision_score(y,x.regression) if y.nunique()==2 else np.nan,AUROC=roc_auc_score(y,x.regression) if y.nunique()==2 else np.nan))
   print('DONE',seed,task,flush=True)
 frames={'applicability_by_seed':pd.DataFrame(rows),'calibration_by_seed':pd.DataFrame(cal),'reliability_bins':pd.DataFrame(reliability),'selective_risk_by_seed':pd.DataFrame(risk),'target_concentration_by_seed':pd.DataFrame(tail),'magnitude_thresholds_by_seed':pd.DataFrame(threshold),'query_support_audit':pd.concat(enriched,ignore_index=True)}
 for name,f in frames.items():f.to_csv(OUT/f'{name}.csv',index=False)
 draw(frames)
 lines=['# Frozen-model additional analyses','','Six models, all five original scaffold splits. Models and query outcomes are unchanged. Strong controls are included regardless of their mean ranking. These are post-hoc descriptive diagnostics, not independent validation or causal experiments. SD reflects dependent scaffold evaluations, not confidence intervals.','','## Probability calibration','ECE uses ten equal-width bins; Brier and log loss are evaluated without recalibration on test data.','']
 summary=frames['calibration_by_seed'].groupby(['task','model'])[['Brier','log_loss','ECE10']].mean();lines.append(summary.round(4).to_string())
 lines+=['','## Applicability domain','Nearest-reference Tanimoto uses the exact training-only reference manifest capped at100. Task1 uses the smaller of the two query molecules’ nearest similarities. Bin membership is fixed without outcome access. AUROC requires both classes and pair AUROC retains the original >=10 queries / >=3 per class rule. Empty/one-class bins remain missing.','',frames['applicability_by_seed'].query('model=="context_neural"').groupby(['task','sim_bin'])[['n','AP','AUROC','macro_MAE']].mean().round(4).to_string(),'', '## Target concentration','The top10 targets are defined per split from the capped training-reference pool. Removing their test queries is a robustness diagnostic, not target-disjoint training.','',frames['target_concentration_by_seed'].query('model=="context_neural"').groupby(['task','subset'])[['n','targets','AP','pair_macro_AUROC','macro_MAE']].mean().round(4).to_string(),'', '## Selective prediction','Retention orders are fixed decision entropy, posterior signed-shift/contrast variance, and one deterministic random ordering per seed. MAE is micro to keep within-retained-sample interpretation. For Task2 the variance describes signed shift, not calibrated absolute sensitivity; low variance need not imply low magnitude error. Filtering changes pair composition and class prevalence and is not a matched comparison to full coverage. Low classification error may partly reflect enrichment of negative/easy queries, so it does not by itself establish calibration or within-pair utility.','',frames['selective_risk_by_seed'].groupby(['task','ordering','coverage'])[['positive_rate','classification_error','micro_MAE']].mean().round(4).to_string(),'', '## Sensitivity thresholds','Task2 continuous magnitude output is ranked at predeclared0.3,0.5,1.0,1.5 pIC50 thresholds. This tests regression ranking across severities, not a classifier trained/calibrated for each threshold.','',frames['magnitude_thresholds_by_seed'].query('model=="context_neural"').groupby('threshold')[['positive','AP','AUROC']].mean().round(4).to_string(),'', '## Next controlled interventions','Reference-count dose response with frozen weights, matched random seeds and nested supports; matched reference-quality interventions at constant count; train-only permutation null with repeated refits; genuinely held-out target/source or Ki replication. Separate nested-support and reference-response permutation intervention results are recorded under reference_interventions/; refitting permutation nulls and external replication remain outstanding.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (OUT/'manifest.json').write_text(json.dumps({'seeds':SEEDS,'models':MODELS,'training_runs':0,'protocol':'frozen-model post-hoc diagnostics; no test recalibration or tuning; all outcomes retained','sources_sha256':SOURCES,'structure_features_sha256':hashlib.sha256((DATA/'protected_decision_gp_20261009/structure_features.npz').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 print(summary.round(4).to_string(),flush=True)

def draw(f):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':INK,'axes.linewidth':.7,'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False,'svg.fonttype':'none'})
 fig,ax=plt.subplots(4,2,figsize=(9,11));letters=iter('abcdefgh')
 for t in [1,2]:
  j=t-1
  for model in MODELS:
   g=f['applicability_by_seed'].query('task==@t and model==@model').groupby('sim_bin').AUROC.agg(['mean','std']).reindex(SIM_LABELS)
   ax[0,j].errorbar(np.arange(5),g['mean'],yerr=g['std'],color=COLORS[model],marker='o',markersize=3,lw=1,label=LABELS[model])
   g=f['reliability_bins'].query('task==@t and model==@model');g=g[g.n>0].groupby('bin')[['mean_probability','observed_rate']].mean()
   ax[1,j].plot(g.mean_probability,g.observed_rate,'o-',color=COLORS[model],ms=3,lw=1)
  ax[0,j].set(xticks=np.arange(5),xticklabels=SIM_LABELS,xlabel='Nearest training-reference similarity',ylabel='AUROC');ax[0,j].tick_params(axis='x',labelsize=8);ax[0,j].set_ylim(.45,1.01)
  ours=f['applicability_by_seed'].query('task==@t and model=="context_neural"').groupby('sim_bin').n.mean().reindex(SIM_LABELS)
  ax[0,j].text(.02,1.07,'Mean bin n: '+', '.join(f'{v:.1f}' for v in ours),transform=ax[0,j].transAxes,fontsize=8)
  ax[1,j].plot([0,1],[0,1],'--',color='#ADB5BD',lw=.8,zorder=0);ax[1,j].set(xlabel='Predicted probability',ylabel='Observed event frequency',xlim=(0,1),ylim=(0,1))
  for order,col in [('decision_entropy',RED),('posterior_variance',BLUE),('random',PEACH)]:
   g=f['selective_risk_by_seed'].query('task==@t and ordering==@order').groupby('coverage').micro_MAE.agg(['mean','std'])
   ax[2,j].errorbar(g.index,g['mean'],yerr=g['std'],color=col,marker='o',ms=3,lw=1,label=order.replace('_',' ').capitalize())
  ax[2,j].set(xlabel='Retained fraction',ylabel='Retained-sample MAE (pIC50)')
  tail=f['target_concentration_by_seed'].query('task==@t');
  for model in MODELS:
   g=tail[tail.model.eq(model)].groupby('subset').pair_macro_AUROC.agg(['mean','std']).reindex(['all','exclude_top10','top10'])
   ax[3,j].errorbar([0,1,2],g['mean'],yerr=g['std'],color=COLORS[model],marker='o',ms=3,lw=1)
  ax[3,j].set(xticks=[0,1,2],xticklabels=['All','Without top 10','Top 10 only'],ylabel='Within-pair AUROC')
 for a in ax.flat:a.text(-.06,1.09,next(letters),transform=a.transAxes,fontweight='bold',fontsize=13);a.grid(axis='y',alpha=.2,color=GRAY,lw=.6,zorder=0)
 handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.005),fontsize=9)
 ax[2,0].legend(fontsize=8,loc='best');fig.text(.29,.943,'Task 1 | SAR transfer',ha='center',fontsize=10);fig.text(.78,.943,'Task 2 | Molecular sensitivity',ha='center',fontsize=10)
 fig.tight_layout(rect=[0,0,1,.93],h_pad=2)
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'additional_diagnostics.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
