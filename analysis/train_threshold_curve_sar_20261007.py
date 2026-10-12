"""Single threshold-conditioned classifier with monotone retention-risk curve.

Training label at threshold t: robust signed assay-B gap < t. The original
task is recovered at t=0.3. Repeated threshold supervision adds no independent
molecules; weights sum to one per original comparison. No model ensemble.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from prepare_sar_transfer_20261007 import OUT, SEEDS, digest
from train_sar_transfer_baseline_20261007 import read, make_molecule_features, assay_features
from train_anchored_sar_transfer_20261007 import encode, metrics
from train_contrast_boost_sar_20261007 import features, pair_history

RESULTS=OUT/'models_threshold_curve_20261007'
RESULTS.mkdir(exist_ok=True)
THRESHOLDS=np.array([-.3,0.,.3,.6,1.0])
THRESHOLD_WEIGHTS=np.array([.125,.125,.5,.125,.125])


def with_threshold(x,threshold):
    return np.column_stack([x,np.full(len(x),threshold,dtype=np.float32)]).astype(np.float32)


def main(seeds):
    data=read('anchor_transfer_all.parquet')
    molecules=make_molecule_features(set(data.identity_1)|set(data.identity_2))
    results=[];audit=[]
    for seed in seeds:
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        parts={part:data.merge(split.loc[split.partition.eq(part),['comparison_id']],on='comparison_id',validate='one_to_one')
               for part in ['train','validation','test']}
        tr=parts['train']
        train_sc=set(tr.scaffold_1)|set(tr.scaffold_2)
        for part in ['validation','test']:
            assert train_sc.isdisjoint(set(parts[part].scaffold_1)|set(parts[part].scaffold_2))
        assays,med,tfidf,svd,formats=assay_features(tr,pd.concat(parts.values()))
        targets={i:np.zeros(1,dtype=np.float32) for i in data.target_chembl_id.unique()}
        base=joblib.load(OUT/'models_contrast_boost_20261007'/f'preprocessors_seed_{seed}.joblib')
        x={}
        for part,frame in parts.items():
            blocks=encode(frame,molecules,assays,med,targets)
            history=pair_history(tr,frame,loo=part=='train')
            x[part]=features(frame,blocks,history,base['variant'])
        margin=np.where(tr.gap_A.to_numpy()>0,tr.min_B_1-tr.max_B_2,tr.min_B_2-tr.max_B_1)
        assert np.array_equal(margin<.3-1e-10,tr.binary_loss.to_numpy().astype(bool))
        expanded=np.concatenate([with_threshold(x['train'],t) for t in THRESHOLDS])
        y=np.concatenate([(margin<t-1e-10).astype(np.int8) for t in THRESHOLDS])
        weights=np.repeat(THRESHOLD_WEIGHTS,len(tr)).astype(np.float32)
        assert np.isclose(weights.sum(),len(tr))
        val=with_threshold(x['validation'],.3)
        joblib.dump({**base,'tfidf':tfidf,'assay_svd':svd,'assay_formats':formats,'numeric_median':med},
                    RESULTS/f'preprocessors_seed_{seed}.joblib',compress=3)
        for monotone in [False,True]:
            name='threshold_monotone' if monotone else 'threshold_unconstrained'
            constraints=tuple([0]*x['train'].shape[1]+[int(monotone)])
            model=XGBClassifier(n_estimators=600,max_depth=base['depth'],learning_rate=.04,
                                min_child_weight=20,subsample=.85,colsample_bytree=.85,
                                reg_lambda=10,tree_method='hist',n_jobs=8,random_state=seed,
                                objective='binary:logistic',eval_metric='logloss',
                                early_stopping_rounds=35,monotone_constraints=constraints)
            start=time.monotonic()
            model.fit(expanded,y,sample_weight=weights,
                      eval_set=[(val,parts['validation'].binary_loss)],verbose=False)
            for part in ['validation','test']:
                pred=model.predict_proba(with_threshold(x[part],.3))[:,1]
                results.append({'seed':seed,'model':name,'partition':part,
                                'best_iteration':model.best_iteration,**metrics(parts[part],pred)})
                output=parts[part][['comparison_id','pair_id','target_chembl_id','binary_loss']].copy()
                output['prediction']=pred
                output.to_csv(RESULTS/f'predictions_{name}_{part}_seed_{seed}.csv',index=False)
            grid=np.array([-.3,0,.15,.3,.45,.6,.8,1.0])
            risks=np.stack([model.predict_proba(with_threshold(x['test'],t))[:,1] for t in grid],axis=1)
            violations=(np.diff(risks,axis=1)<-1e-7).any(axis=1)
            if monotone:assert not violations.any()
            audit.append({'seed':seed,'model':name,'test_n':len(risks),
                          'nonmonotone_fraction':float(violations.mean())})
            curve=pd.DataFrame(risks,columns=[f'risk_threshold_{t:g}' for t in grid])
            curve.insert(0,'comparison_id',parts['test'].comparison_id.to_numpy())
            curve.to_csv(RESULTS/f'curve_{name}_seed_{seed}.csv',index=False)
            joblib.dump(model,RESULTS/f'{name}_seed_{seed}.joblib',compress=3)
            pd.DataFrame(results).to_csv(RESULTS/'metrics.csv',index=False)
            print(seed,name,'seconds',round(time.monotonic()-start,1),'test',results[-1],flush=True)
        del expanded,x
    pd.DataFrame(audit).to_csv(RESULTS/'monotonicity_audit.csv',index=False)
    (RESULTS/'manifest.json').write_text(json.dumps({
        'method':'One threshold-conditioned boosted tree classifier; optional nondecreasing threshold constraint',
        'thresholds':THRESHOLDS.tolist(),'threshold_weights':THRESHOLD_WEIGHTS.tolist(),
        'evaluation':'Original unchanged binary_loss at threshold 0.3; training B values only used as targets',
        'model_parameters':'Depth and pair-history inclusion inherited from validation-selected single booster; validation early stopping',
        'seeds':seeds,'source_sha256':digest(OUT/'anchor_transfer_all.parquet'),
        'code_sha256':digest(Path(__file__)),'xgboost_version':xgboost.__version__},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    main(parser.parse_args().seeds)
