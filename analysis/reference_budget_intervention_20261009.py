"""Frozen-weight nested reference-count and reference-response interventions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from train_contextual_decision_gp_20261009 import cohorts,infer,RESULTS,ContextualDecisionGP,feature_cache,read,DEVICE,SEEDS
from make_final_baseline_tables_20261009 import scores,DATA
OUT=DATA/'additional_frozen_analyses_20261009/reference_interventions'
def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 obs=read('molecule_assay_observations.parquet');anchor=read('anchor_transfer_all.parquet');mapping,bits,desc=feature_cache();allrows=pd.read_csv(OUT/'metrics_by_seed.csv').to_dict('records') if (OUT/'metrics_by_seed.csv').exists() else [];audit=pd.read_csv(OUT/'reference_audit.csv').to_dict('records') if (OUT/'reference_audit.csv').exists() else []
 for seed in SEEDS:
  _,_,_,_,_,frames=cohorts(seed,obs,anchor)
  ref=pd.read_csv(DATA/'controlled_joint_ablation_20261009'/f'reference_manifest_seed_{seed}.csv');lookup=obs.set_index('sample_id');original_groups={p:lookup.loc[g.sample_id].reset_index() for p,g in ref.groupby('pair_id',sort=False)};groups={p:g.sample(frac=1,random_state=seed+97).reset_index(drop=True) for p,g in original_groups.items()}
  cp_path=RESULTS/f'context_neural_selected_seed_{seed}.pt';cp=torch.load(cp_path,map_location=DEVICE,weights_only=False)
  matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE);model=ContextualDecisionGP(cp['dim'],'context_neural').to(DEVICE);model.load_state_dict(cp['state_dict']);model.eval()
  for condition,budget in [('nested',3),('nested',10),('nested',30),('nested',100),('shuffle_shift',100)]:
   if sum(r['seed']==seed and r['condition']==condition and r['budget']==budget for r in allrows)==2:continue
   chosen={p:g.iloc[:budget].copy() for p,g in (original_groups if budget==100 else groups).items()}
   if condition=='shuffle_shift':
    for p,g in chosen.items():
     salt=int(hashlib.sha256((p+str(seed)).encode()).hexdigest()[:8],16);g['signed_delta']=np.random.default_rng(salt).permutation(g.signed_delta.to_numpy())
   for t in [1,2]:
    pred=infer(model,frames['test'][t],chosen,mapping,matrix,t);key='comparison_id' if t==1 else 'sample_id'
    original=pd.read_csv(RESULTS/f'task{t}_context_neural_selected_test_seed_{seed}.csv');assert np.array_equal(pred[key],original[key]);assert np.array_equal(pred.binary_loss,original.binary_loss);assert np.allclose(pred.target,original.target,atol=1e-8)
    if condition=='nested' and budget==100:
     diff=np.max(abs(pred.prediction-original.prediction));assert diff<2e-4,(seed,t,diff)
    pred.to_csv(OUT/f'task{t}_{condition}_k{budget}_seed{seed}.csv',index=False)
    allrows.append(dict(seed=seed,task=t,condition=condition,budget=budget,**scores(pred)))
   used=set(frames['test'][1].pair_id)|set(frames['test'][2].pair_id)
   audit.append(dict(seed=seed,condition=condition,budget=budget,min_actual_references=min(len(chosen[p]) for p in used),mean_actual_references=np.mean([len(chosen[p]) for p in used]),max_actual_references=max(len(chosen[p]) for p in used),checkpoint_sha256=hashlib.sha256(cp_path.read_bytes()).hexdigest(),inference_device=DEVICE))
   pd.DataFrame(allrows).to_csv(OUT/'metrics_by_seed.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'reference_audit.csv',index=False);print('INTERVENTION',seed,condition,budget,flush=True)
 summary=pd.DataFrame(allrows).groupby(['task','condition','budget'])[['AP','pair_macro_AUROC','macro_MAE']].agg(['mean','std']);summary.to_csv(OUT/'summary.csv');print(summary.to_string(),flush=True)
 (OUT/'REPORT.md').write_text('# Frozen reference interventions\n\nFixed five splits, selected model weights, test queries and labels. No retraining or selection on these results. References use a deterministic shuffled ordering of the original capped training-only support, with nested sets3/10/30/100. At100 the original matrix ordering is restored to reproduce the exact original numerical computation; all smaller supports are subsets of the100pool. The100reference predictions reproduce original predictions to tolerance2e-4. Groups with fewer observations retain all available references. Shuffle control permutes signed responses among historical compounds within each assay pair at the same100cap, retaining the marginal shift distribution and source potencies. This is a fitted-model information intervention, not a training permutation null or a causal biological experiment.\n\nTask2 training-only pair median and labels stay fixed across reference counts; they still use the original full training pool. Therefore this probes posterior/adaptor reference dependence, not end-to-end learning from onlykmeasurements. Task1 support-derived replicate width changes with the selected references. SD summarizes dependent splits, not confidence intervals. Only one deterministic support/shuffle realization per split was run; reference-selection stability is unmeasured.\n\n'+summary.round(4).to_string()+'\n')
if __name__=='__main__':main()
