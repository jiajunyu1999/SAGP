"""Audit risk curves on unchanged held-out rows and all reported thresholds."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from prepare_sar_transfer_20261007 import OUT,SEEDS
from train_sar_transfer_baseline_20261007 import read


def main():
    root=OUT/'models_threshold_curve_20261007'
    data=read('anchor_transfer_all.parquet')
    rows=[]
    for seed in SEEDS:
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        train=data.merge(split.loc[split.partition.eq('train'),['comparison_id']],on='comparison_id')
        test=data.merge(split.loc[split.partition.eq('test'),['comparison_id']],on='comparison_id')
        def margin(frame):
            return np.where(frame.gap_A>0,frame.min_B_1-frame.max_B_2,frame.min_B_2-frame.max_B_1)
        train_margin=margin(train);test_margin=margin(test)
        for model in ['threshold_unconstrained','threshold_monotone']:
            curve=pd.read_csv(root/f'curve_{model}_seed_{seed}.csv')
            assert curve.comparison_id.equals(test.comparison_id)
            risk=curve.drop(columns='comparison_id')
            if model=='threshold_monotone':assert (np.diff(risk.to_numpy(),axis=1)>=-1e-7).all()
            for col in risk:
                threshold=float(col.split('risk_threshold_')[1])
                train_y=(train_margin<threshold-1e-10).astype(int)
                yy=(test_margin<threshold-1e-10).astype(int)
                stats=train.assign(y=train_y).groupby('pair_id').y.agg(['sum','count'])
                prior=(test.pair_id.map(stats['sum'])+20*train_y.mean())/(test.pair_id.map(stats['count'])+20)
                if abs(threshold-.3)<1e-10:assert np.array_equal(yy,test.binary_loss)
                rows.append({'seed':seed,'model':model,'threshold':threshold,'test_n':len(test),
                             'positive_rate':yy.mean(),'Brier':brier_score_loss(yy,risk[col]),
                             'pair_prior_Brier':brier_score_loss(yy,prior)})
    result=pd.DataFrame(rows);result.to_csv(root/'curve_evaluation.csv',index=False)
    scores=pd.read_csv(root/'metrics.csv').query("partition=='test'")
    baseline=pd.read_csv(OUT/'models_contrast_boost_20261007'/'metrics.csv').query("partition=='test'").set_index('seed')
    summary={}
    for model,frame in scores.groupby('model'):
        frame=frame.set_index('seed')
        summary[model]={
            'means':frame[['AP','AUROC','pair_macro_AUROC','pair_macro_AP','Brier']].mean().to_dict(),
            'delta_vs_single_booster':(frame[['AP','AUROC','pair_macro_AUROC']]-baseline[['AP','AUROC','pair_macro_AUROC']]).mean().to_dict(),
            'AP_improved_seeds':int((frame.AP>baseline.AP).sum()),
            'curve_Brier':float(result.loc[result.model.eq(model),'Brier'].mean()),
            'curve_prior_Brier':float(result.loc[result.model.eq(model),'pair_prior_Brier'].mean())}
    (root/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
