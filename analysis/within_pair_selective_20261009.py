"""Keep every assay pair represented at each budget; score controls on identical queries."""
import numpy as np
import pandas as pd
from additional_frozen_analyses_20261009 import OUT,DATA,SEEDS,MODELS,known_models,load
rows=[]
for seed in SEEDS:
 for task in [1,2]:
  key='comparison_id' if task==1 else 'sample_id'
  a=pd.read_csv(DATA/'contextual_decision_gp_20261009'/f'task{task}_context_neural_selected_test_seed_{seed}.csv')
  controls={}
  for folder,model,_ in known_models(task):
   if model not in MODELS:continue
   filename=f'task{task}_{model}_selected_test_seed_{seed}.csv' if model in ['context_neural','detached_action'] else f'task{task}_{model}_test_seed_{seed}.csv'
   x=load(DATA/folder/filename).rename(columns={'gap_B_target':'target','gap_B_mean':'regression'}).set_index(key).loc[a[key]].reset_index();assert np.array_equal(a.binary_loss,x.binary_loss);controls[model]=x
  p=np.clip(a.prediction,1e-7,1-1e-7).to_numpy();orderings={'decision_entropy':-(p*np.log(p)+(1-p)*np.log1p(-p)),'posterior_variance':a.variance.to_numpy(),'random':np.random.default_rng(seed+task).random(len(a))}
  for ordering,values in orderings.items():
   for budget in [.2,.4,.6,.8,1.]:
    selected=[]
    for _,idx in a.groupby('pair_id',sort=False).indices.items():selected.extend(idx[np.argsort(values[idx],kind='stable')[:max(1,int(np.ceil(len(idx)*budget)))]])
    for model,x in controls.items():
     g=x.iloc[selected].copy();g['error']=abs(g.target-g.regression);g['classification_error']=((g.prediction>=.5).astype(int)!=g.binary_loss).astype(float)
     rows.append(dict(seed=seed,task=task,model=model,ordering=ordering,budget=budget,n=len(g),actual_fraction=len(g)/len(x),pairs=g.pair_id.nunique(),positive_rate=g.binary_loss.mean(),pair_macro_MAE=g.groupby('pair_id').error.mean().mean(),pair_macro_classification_error=g.groupby('pair_id').classification_error.mean().mean()))
d=pd.DataFrame(rows);d.to_csv(OUT/'within_pair_selective_by_seed.csv',index=False)
s=d.groupby(['task','model','ordering','budget'])[['actual_fraction','positive_rate','pair_macro_MAE','pair_macro_classification_error']].mean();s.to_csv(OUT/'within_pair_selective_summary.csv')
(OUT/'within_pair_selective_REPORT.md').write_text('# Pair-stratified selective prediction\n\nRetain the least uncertain ceil(fraction*n_pair) queries from every assay pair, at least one per pair. Every model is scored on identical selected queries. Confidence/variance ordering uses the frozen Decision GP, never test outcomes. All pairs are retained with equal macro weighting, controlling pair-composition changes in global retention. Positive prevalence still changes with query selection. This is descriptive and uses five dependent original splits; it is not prospective calibration or evidence of unseen-target generalization.\n\n'+s.round(4).to_string()+'\n')
print(s.xs('context_neural',level='model').round(4).to_string())
