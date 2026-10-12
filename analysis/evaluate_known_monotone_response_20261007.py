"""Frozen known-pair field evaluation with MAE-consistent Gaussian actions."""
import json,multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,DEVICE,SEEDS
from prepare_sar_transfer_20261007 import OUT,partition
from monotone_response_gp_20261007 import MonotoneResponseGP
from train_known_monotone_response_20261007 import RESULTS,infer
from train_known_response_field_20261007 import metric
from response_distribution_decisions_20261007 import recover_mu_absolute,folded_median
from neural_transition_profiles_20261007 import reference_profiles
from audit_response_field_20261007 import separation
from summarize_deep_three_tasks_20261007 import bootstrap
from deep_three_tasks_20261007 import RESULTS as DEEP


def main():
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=pd.read_csv(RESULTS/'metrics.csv').query("partition!='test'").to_dict('records')
    for seed in SEEDS:
        sp=obs.scaffold.map(lambda s:partition(s,seed));train=obs[sp.eq('train')]
        groups={p:g for p,g in train.groupby('pair_id')};center=train.groupby('pair_id').signed_delta.median()
        known=train.groupby('pair_id').size();known=set(known[known>=30].index)
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet');sar=data.merge(split[split.partition.eq('test')][['comparison_id']],on='comparison_id')
        sens=obs[sp.eq('test')&obs.pair_id.isin(known)].copy().reset_index(drop=True)
        sens['pair_offset']=sens.pair_id.map(center);sens['residual_abs']=abs(sens.signed_delta-sens.pair_offset);sens['binary_loss']=(sens.residual_abs>=.5).astype(int)
        separation(train,sar,1,seed,'test');separation(train,sens,2,seed,'test')
        profile=reference_profiles({'test':sens},obs,seed,2)['test']
        baseline=sens[['sample_id','pair_id','target_chembl_id','binary_loss']].copy();baseline['prediction']=profile['risk'];baseline['target']=sens.residual_abs
        baseline['regression']=folded_median(profile['features'][:,-2],profile['features'][:,7]);baseline['variance']=profile['features'][:,7]**2
        baseline.to_csv(RESULTS/f'task2_reference_gaussian_median_test_seed_{seed}.csv',index=False)
        rows.append({'seed':seed,'task':2,'model':'reference_gaussian_median','partition':'test',**metric(sens,baseline,2)})
        for name in ['global_monotone','conditional_monotone']:
            cp=torch.load(RESULTS/f'{name}_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
            model=MonotoneResponseGP(cp['dim'],name=='conditional_monotone').to(DEVICE);model.load_state_dict(cp['state_dict']);model.eval()
            matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
            for task,f in [(1,sar),(2,sens)]:
                pred=infer(model,f,groups,mapping,matrix,task)
                if task==2:
                    mu,sd=recover_mu_absolute(pred.regression,pred.variance);pred['expected_absolute']=pred.regression;pred['regression']=folded_median(mu,sd)
                pred.to_csv(RESULTS/f'task{task}_{name}_test_seed_{seed}.csv',index=False)
                rows.append({'seed':seed,'task':task,'model':name,'partition':'test','best_step':cp['best_step'],'readout':'median_action',**metric(f,pred,task)})
            pd.DataFrame(rows).to_csv(RESULTS/'metrics.csv',index=False)
            print(seed,name,pd.DataFrame(rows).query('seed==@seed and model==@name and partition=="test"')[['task','AP','pair_macro_AUROC','macro_MAE']].round(4).to_dict('records'),flush=True)
    test=pd.DataFrame(rows).query('partition=="test"');summary=test.groupby(['task','model'])[['AP','AUROC','pair_macro_AUROC','macro_MAE']].mean()
    summary.to_csv(RESULTS/'test_summary.csv');print(summary.round(5).to_string(),flush=True)
    jobs=[]
    for task,base in [(1,'threshold_monotone'),(1,'global_monotone'),(1,'fixed_tanimoto_gp'),
        (2,'local_knn'),(2,'reference_gaussian_median'),(2,'global_monotone'),(2,'calibrated_attention')]:
        records=[]
        for seed in SEEDS:
            a=pd.read_csv(RESULTS/f'task{task}_conditional_monotone_test_seed_{seed}.csv')
            if base=='threshold_monotone':
                b=pd.read_csv(OUT/'models_threshold_curve_20261007'/f'predictions_threshold_monotone_test_seed_{seed}.csv')
                reg=pd.read_csv(DEEP/f'task1_contrast_boost_with_gap_regression_test_seed_{seed}.csv').set_index('comparison_id')
                b['target']=reg.loc[b.comparison_id,'target'].to_numpy();b['regression']=reg.loc[b.comparison_id,'regression'].to_numpy()
            else:
                directory=RESULTS if base in ['global_monotone','reference_gaussian_median'] else OUT/'models_known_response_field_20261007' if base=='fixed_tanimoto_gp' else DEEP
                b=pd.read_csv(directory/f'task{task}_{base}_test_seed_{seed}.csv')
            key='comparison_id' if task==1 else 'sample_id';b=b.set_index(key).loc[a[key]].reset_index();records.append((a,b))
        jobs.append((task,base,records))
    comparisons=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(bootstrap,r,roots) for _,_,r in jobs]
        for (task,base,_),future in zip(jobs,futures):
            r=future.result();r.update(task=task,neural='conditional_monotone',baseline=base);comparisons.append(r);print(json.dumps(r),flush=True)
    (RESULTS/'test_source_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    manifest=json.loads((RESULTS/'manifest.json').read_text());manifest.update(test_evaluated=True,completed_seeds=SEEDS,
        decisions='Absolute-residual median,not expectation,for both probabilistic neural field and Gaussian calibration control;AP scores unchanged',
        selection='All checkpoints selected by validation root shift NLL;fixed candidate architectures;no test selection')
    (RESULTS/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
