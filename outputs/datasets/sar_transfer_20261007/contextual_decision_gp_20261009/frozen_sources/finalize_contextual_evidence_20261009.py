"""Additional frozen-control comparisons and artifact provenance, no refitting."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from train_contextual_decision_gp_20261009 import RESULTS,KINDS
from train_protected_decision_gp_20261009 import RESULTS as GLOBAL
from controlled_joint_ablation_20261009 import RESULTS as CONTROL
from meta_gp_sar_20261007 import read,context_components,SEEDS
from summarize_controlled_joint_ablation_20261009 import prepare,contrasts
from make_expanded_baseline_tables_20261008 import known_models,DATA,align
from contextual_decision_gp_20261009 import ContextualDecisionGP


def main():
    chosen=json.loads((RESULTS/'selection.json').read_text())['selected_model']
    roots=context_components(read('molecule_assay_observations.parquet'));lookup=json.loads((CONTROL/'bootstrap_components.json').read_text());draws=np.load(CONTROL/'bootstrap_component_weights.npy')
    rows=[]
    for ck in ['selected','final']:
        for t,controls in [(1,['threshold_monotone','fixed_tanimoto_gp','detached_action']),(2,['local_knn','fixed_tanimoto_gp','detached_action'])]:
            for control in controls:
                packed=[]
                for seed in SEEDS:
                    a=pd.read_csv(RESULTS/f'task{t}_{chosen}_{ck}_test_seed_{seed}.csv')
                    if control=='detached_action':p=GLOBAL/f'task{t}_detached_action_{ck}_test_seed_{seed}.csv'
                    else:
                        folder=next(d for d,n,_ in known_models(t) if n==control)
                        name=f'predictions_{control}_test_seed_{seed}.csv' if control=='threshold_monotone' else f'task{t}_{control}_test_seed_{seed}.csv'
                        p=DATA/folder/name
                    b=pd.read_csv(p).rename(columns={'gap_B_target':'target','gap_B_mean':'regression'})
                    b=align(b,a,t)
                    if 'regression' not in b:
                        # This binary baseline has no continuous estimator.
                        # Populate a sentinel solely to share the bootstrap engine;
                        # its meaningless error contrast is discarded below.
                        assert control=='threshold_monotone'
                        b['regression']=a.regression.to_numpy();b['target']=a.target.to_numpy()
                    packed.append(prepare(a,b,roots,lookup))
                values=contrasts(packed,draws)
                for j,k in enumerate(['AP','pair_macro_AUROC','macro_MAE']):
                    if control=='threshold_monotone' and k=='macro_MAE':continue
                    valid=values[1:,j][np.isfinite(values[1:,j])];lo,hi=np.quantile(valid,[.025,.975])
                    rows.append(dict(checkpoint=ck,task=t,control=control,metric=k,new_minus_control=values[0,j],CI_low=lo,CI_high=hi,evaluable_draws=len(valid)))
                print('EXTRA BOOTSTRAP',ck,t,control,flush=True)
    base=pd.read_csv(RESULTS/'paired_component_bootstrap.csv');pd.concat([base,pd.DataFrame(rows)],ignore_index=True).to_csv(RESULTS/'all_paired_intervals.csv',index=False)
    frozen=RESULTS/'frozen_sources';frozen.mkdir(exist_ok=True)
    for name in ['contextual_decision_gp_20261009.py','train_contextual_decision_gp_20261009.py','evaluate_contextual_decision_gp_20261009.py','finalize_contextual_evidence_20261009.py']:
        shutil.copy2(Path('analysis')/name,frozen/name)
    audits=[]
    for seed in SEEDS:
        old=json.loads((CONTROL/f'control_audit_seed_{seed}.json').read_text());done=json.loads((CONTROL/f'complete_joint_seed_{seed}.json').read_text())
        for k in KINDS:
            r=json.loads((RESULTS/f'complete_{k}_seed_{seed}.json').read_text())
            assert r['batch_sha256']==done['batch_sha256'] and r['pretrain_sha256']==old['pretrain_sha256'] and r['steps']==1000
            actual=sum(p.numel() for p in ContextualDecisionGP(2059,k).parameters())
            audits.append({**r,'parameters_reported_in_raw_log':r['parameters'],'parameters':actual,'decision_parameters':actual-1483})
    pd.DataFrame(audits).to_csv(RESULTS/'compute_and_control_audit.csv',index=False)
    (RESULTS/'metadata_correction.json').write_text(json.dumps({'raw_fit_logs':'The copied parameter-count field1494 is stale; raw records are preserved unchanged. Architecture-based counts are1539 for linear and1883 for neural. Training/checkpoints/predictions are unaffected.','corrected_audit':'compute_and_control_audit.csv'},indent=2))
    print('AUDIT PASSED',flush=True)


if __name__=='__main__':main()
