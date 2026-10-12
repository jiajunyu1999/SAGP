"""Reload each released SAGP checkpoint; check published selected-query predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from train_contextual_decision_gp_20261009 import feature_cache,cohorts,evaluate,RESULTS,SEEDS,read,DEVICE
from contextual_decision_gp_20261009 import ContextualDecisionGP
B=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);mapping,bits,desc=feature_cache();obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');rows=[]
for seed in SEEDS:
 cp=torch.load(RESULTS/f'context_neural_selected_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
 model=ContextualDecisionGP(cp['dim'],'context_neural').to(DEVICE);model.load_state_dict(cp['state_dict'])
 assert sum(p.numel() for p in model.parameters())==1883
 matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
 _,groups,_,_,_,frames=cohorts(seed,obs,data);_,metrics,pred=evaluate(model,frames['test'],groups,mapping,matrix)
 for task in [1,2]:
  key='comparison_id' if task==1 else 'sample_id';original=pd.read_csv(RESULTS/f'task{task}_context_neural_selected_test_seed_{seed}.csv').set_index(key).loc[pred[task][key]]
  assert np.array_equal(original.binary_loss.to_numpy(),pred[task].binary_loss.to_numpy())
  delta={k:float(np.max(np.abs(original[k].to_numpy()-pred[task][k].to_numpy()))) for k in ['prediction','regression','target']}
  assert max(delta.values())<2e-4,(seed,task,delta)
  rows.append({'seed':seed,'task':task,'n':len(original),'maximum_absolute_difference':delta,'recomputed_metrics':metrics[task]})
 print('Verified SAGP seed',seed,flush=True)
(B/'verification').mkdir(exist_ok=True);(B/'verification/checkpoint_predictions.json').write_text(json.dumps(rows,indent=2))
