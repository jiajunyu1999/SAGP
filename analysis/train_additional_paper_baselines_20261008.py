"""Fixed linear and random-forest controls on the original task 1/2 splits.

Reuse the deep controls' train-only feature construction without retrieving
reference edges. Separate classification/regression estimators serve distinct
endpoints; no fitted predictions are fused. Hyperparameters are fixed before
test evaluation, and the held-out assay-B values are labels only.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits
from deep_three_tasks_20261007 import (
    prepare_task1, prepare_task2, audit, eval_metrics, read,
    make_molecule_features, OUT, SEEDS,
)

RESULTS=OUT/'models_additional_paper_baselines_20261008'


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    data=read('anchor_transfer_all.parquet')
    obs=read('molecule_assay_observations.parquet')
    molecules=make_molecule_features(set(obs.chemical_identity)|set(data.identity_1)|set(data.identity_2))
    rows=[]
    for task in [1,2]:
        for seed in SEEDS:
            print(f'Preparing task {task}, seed {seed}',flush=True)
            prepare=prepare_task1 if task==1 else prepare_task2
            frames,arrays,_=prepare(seed,data if task==1 else obs,molecules,include_references=False)
            audit(frames,task)
            tr=arrays['train']
            candidates=[
                ('linear', LogisticRegression(C=1.,solver='lbfgs',max_iter=1500,random_state=seed),Ridge(alpha=10.)),
                ('random_forest',RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=10,max_features='sqrt',n_jobs=4,random_state=seed),
                 RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=10,max_features='sqrt',n_jobs=4,random_state=seed)),
            ]
            for name,clf,reg in candidates:
                with threadpool_limits(limits=4):
                    clf.fit(tr['x'],tr['y'])
                    reg.fit(tr['x'],tr['target'])
                for part in ['validation','test']:
                    frame=frames[part];a=arrays[part]
                    pp=clf.predict_proba(a['x'])[:,1];rr=reg.predict(a['x'])
                    if task==2:rr=np.maximum(rr,0)
                    key='comparison_id' if task==1 else 'sample_id'
                    out=frame[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                    out['prediction']=pp;out['target']=frame.target;out['regression']=rr
                    out.to_csv(RESULTS/f'task{task}_{name}_{part}_seed_{seed}.csv',index=False)
                    rows.append(dict(task=task,seed=seed,model=name,partition=part,**eval_metrics(task,frame,pp,rr)))
                pd.DataFrame(rows).to_csv(RESULTS/'metrics.csv',index=False)
                print(task,seed,name,{k:round(rows[-1][k],4) for k in ['AP','pair_macro_AUROC','macro_MAE']},flush=True)
    config={'seeds':list(SEEDS),'tasks':[1,2],'feature_pipeline':'deep_three_tasks prepare_task1/2; no reference edge retrieval',
        'linear':{'classification':'LogisticRegression','C':1.,'max_iter':1500,'regression':'Ridge','alpha':10.},
        'random_forest':{'n_estimators':200,'max_depth':12,'min_samples_leaf':10,'max_features':'sqrt'},
        'selection':'Fixed settings, no test hyperparameter selection',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (RESULTS/'manifest.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
