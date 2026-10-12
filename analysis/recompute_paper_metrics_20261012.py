"""Recompute only active two-task benchmark and response-analysis metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from make_final_baseline_tables_20261009 import known_models,scores,align
B=Path(__file__).resolve().parents[1];D=B/'outputs/datasets/sar_transfer_20261007';expected=pd.read_csv(B/'paper/baseline_table_sources_final_20261009/metrics_by_seed.csv');rows=[]
for record in expected.to_dict('records'):
 t=int(record['task']);seed=int(record['seed']);model=record['model'];key='comparison_id' if t==1 else 'sample_id'
 dirname=next(d for d,n,_ in known_models(t) if n==model)
 filename=f'predictions_{model}_test_seed_{seed}.csv' if model=='threshold_monotone' else (f'task{t}_{model}_selected_test_seed_{seed}.csv' if model=='context_neural' else f'task{t}_{model}_test_seed_{seed}.csv')
 f=pd.read_csv(D/dirname/filename).rename(columns={'gap_B_target':'target','gap_B_mean':'regression'})
 canonical=pd.read_csv(D/'contextual_decision_gp_20261009'/f'task{t}_context_neural_selected_test_seed_{seed}.csv');f=align(f,canonical,t)
 result=scores(f,model!='threshold_monotone')
 for metric in ['AP','AUROC','pair_macro_AUROC','macro_MAE']:
  assert np.allclose(result[metric],record[metric],atol=1e-7,rtol=1e-7,equal_nan=True),(t,seed,model,metric,result[metric],record[metric])
 rows.append(dict(task=t,seed=seed,model=model,label='SAGP' if model=='context_neural' else record['label'],**result))
V=B/'verification';V.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(V/'recomputed_metrics_by_seed.csv',index=False)
(V/'metrics_alignment.json').write_text(json.dumps({'rows_verified':len(rows),'tasks':[1,2],'seeds':[17,23,41,71,101],'alignment':'identical query IDs, assay pairs, labels and targets; all four benchmark metrics within 1e-7 of saved paper values'},indent=2));print('Verified',len(rows),'task/model/seed records.')
