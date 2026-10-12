"""Inference-only covariance removal at the SAME trained Joint checkpoint.

Distinct from F, which removes covariance in both training and inference.
This isolates its predictive-variance effect without changing learned means.
"""
import json
import numpy as np
import pandas as pd
import torch
from controlled_joint_ablation_20261009 import RESULTS,ControlledGP,infer,SEEDS
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,DEVICE
from prepare_sar_transfer_20261007 import partition
from train_known_response_field_20261007 import metric
from summarize_controlled_joint_ablation_20261009 import prepare,contrasts


def main():
    torch.set_num_threads(4)
    obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet')
    roots=context_components(obs);mapping,bits,desc=molecule_matrix(obs)
    lookup=json.loads((RESULTS/'bootstrap_components.json').read_text())
    draws=np.load(RESULTS/'bootstrap_component_weights.npy')
    packed={ck:[] for ck in ['selected','final']};metrics=[];effects=[]
    calibration={ck:[] for ck in packed}
    for seed in SEEDS:
        train=obs[obs.scaffold.map({s:partition(s,seed) for s in obs.scaffold.unique()}).eq('train')]
        groups={p:g for p,g in train.groupby('pair_id')}
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        frame=data.merge(split[split.partition.eq('test')][['comparison_id']],on='comparison_id')
        for ck in ['selected','final']:
            cp=torch.load(RESULTS/f'joint_{ck}_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
            matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
            m=ControlledGP(cp['dim'],'no_contrast_covariance').to(DEVICE);m.load_state_dict(cp['state_dict']);m.eval()
            a=pd.read_csv(RESULTS/f'task1_joint_{ck}_test_seed_{seed}.csv')
            b=infer(m,frame,groups,mapping,matrix,1)
            assert a.comparison_id.equals(b.comparison_id)
            assert np.allclose(a.regression,b.regression,rtol=0,atol=1e-6),'Frozen means changed!'
            b.to_csv(RESULTS/f'task1_frozen_independent_variance_{ck}_test_seed_{seed}.csv',index=False)
            # Compare the same exported representation on both sides; pandas
            # writes float32 values with fewer digits than float64 values.
            b=pd.read_csv(RESULTS/f'task1_frozen_independent_variance_{ck}_test_seed_{seed}.csv')
            metrics.append({'seed':seed,'checkpoint':ck,**metric(frame,b,1)})
            e=a[['comparison_id','pair_id']].copy();e['joint_variance']=a.variance;e['independent_variance']=b.variance
            e['covariance']=.5*(b.variance-a.variance)
            e['fraction_variance_cancelled']=1-a.variance/b.variance
            effects.append(e.assign(seed=seed,checkpoint=ck))
            packed[ck].append(prepare(a,b,roots,lookup))
            values={}
            for label,pred in [('joint',a),('independent',b)]:
                error=(pred.target-pred.regression).to_numpy();v=pred.variance.to_numpy()
                z=pd.DataFrame({'pair_id':pred.pair_id,
                    'Brier':(pred.prediction.to_numpy()-pred.binary_loss.to_numpy())**2,
                    'gap_NLL':.5*(np.log(2*np.pi*v)+error**2/v),
                    'coverage_90':(abs(error)<=1.6448536269514722*np.sqrt(v)).astype(float)})
                values[label]=z.groupby('pair_id').mean(numeric_only=True)
            delta=values['joint']-values['independent']
            pc=np.array([lookup[roots[p]] for p in delta.index])
            calibration[ck].append((pc,delta.to_numpy()))
            print(seed,ck,'frozen means unchanged',flush=True)
    pd.DataFrame(metrics).to_csv(RESULTS/'frozen_covariance_metrics_by_seed.csv',index=False)
    pd.concat(effects,ignore_index=True).to_csv(RESULTS/'frozen_covariance_per_query.csv',index=False)
    rows=[]
    for ck in packed:
        values=contrasts(packed[ck],draws)
        np.save(RESULTS/f'frozen_covariance_bootstrap_{ck}.npy',values)
        for j,key in enumerate(['AP','pair_macro_AUROC','macro_MAE']):
            finite=values[1:,j][np.isfinite(values[1:,j])];lo,hi=np.quantile(finite,[.025,.975])
            rows.append({'checkpoint':ck,'metric':key,'joint_minus_frozen_independent':values[0,j],'CI_low':lo,'CI_high':hi,'evaluable_draws':len(finite)})
    pd.DataFrame(rows).to_csv(RESULTS/'frozen_covariance_paired_intervals.csv',index=False)
    calibration_rows=[]
    for ck in calibration:
        values=[]
        for w in draws:
            runs=[np.average(delta,weights=w[pc],axis=0) if w[pc].sum()>0 else np.full(3,np.nan) for pc,delta in calibration[ck]]
            v=np.array(runs);values.append(v.mean(0))
        values=np.array(values)
        for j,key in enumerate(['Brier','gap_NLL','coverage_90']):
            valid=values[1:,j][np.isfinite(values[1:,j])];lo,hi=np.quantile(valid,[.025,.975])
            calibration_rows.append({'checkpoint':ck,'metric':key,'joint_minus_frozen_independent':values[0,j],'CI_low':lo,'CI_high':hi})
    pd.DataFrame(calibration_rows).to_csv(RESULTS/'frozen_covariance_calibration_intervals.csv',index=False)
    calibration_summary()
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def calibration_summary():
    """Replay saved predictions, including interval width and absolute coverage."""
    roots=context_components(read('molecule_assay_observations.parquet'))
    lookup=json.loads((RESULTS/'bootstrap_components.json').read_text())
    draws=np.load(RESULTS/'bootstrap_component_weights.npy')
    keys=['pair_macro_Brier','pair_macro_gap_NLL','pair_macro_coverage_90','pair_macro_halfwidth_90']
    absolute=[];intervals=[]
    for ck in ['selected','final']:
        packed=[]
        for seed in SEEDS:
            records={}
            for model in ['joint','frozen_independent_variance']:
                f=pd.read_csv(RESULTS/f'task1_{model}_{ck}_test_seed_{seed}.csv')
                err=f.target-f.regression;v=f.variance;half=1.6448536269514722*np.sqrt(v)
                z=pd.DataFrame({'pair_id':f.pair_id,keys[0]:(f.prediction-f.binary_loss)**2,
                    keys[1]:.5*(np.log(2*np.pi*v)+err**2/v),keys[2]:(abs(err)<=half).astype(float),keys[3]:half})
                records[model]=z.groupby('pair_id').mean(numeric_only=True)
                absolute.append({'checkpoint':ck,'seed':seed,'model':model,**records[model].mean().to_dict()})
            delta=records['joint']-records['frozen_independent_variance']
            packed.append((np.array([lookup[roots[p]] for p in delta.index]),delta.to_numpy()))
        samples=[]
        for w in draws:
            runs=[np.average(d,weights=w[pc],axis=0) if w[pc].sum()>0 else np.full(4,np.nan) for pc,d in packed]
            samples.append(np.array(runs).mean(0))
        samples=np.array(samples)
        for j,key in enumerate(keys):
            v=samples[1:,j][np.isfinite(samples[1:,j])];lo,hi=np.quantile(v,[.025,.975])
            intervals.append({'checkpoint':ck,'metric':key,'joint_minus_frozen_independent':samples[0,j],'CI_low':lo,'CI_high':hi})
    pd.DataFrame(absolute).to_csv(RESULTS/'frozen_covariance_calibration_by_seed.csv',index=False)
    pd.DataFrame(intervals).to_csv(RESULTS/'frozen_covariance_calibration_intervals.csv',index=False)


if __name__=='__main__':main()
