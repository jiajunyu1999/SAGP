"""Deployment-aligned assay-A-to-B SAR transfer: eligibility uses assay A only."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from prepare_sar_transfer_20261007 import OUT, SEEDS, PARTS, partition, save, digest


def main():
    con=duckdb.connect()
    pool=con.execute("SELECT * FROM read_parquet(?)",[str(OUT/"all_analogue_comparisons.parquet")]).df()
    data=pool[(pool.tanimoto>=.7)&(pool.shared_molecules>=5)&(pool.robust_gap_A>=.5-1e-10)].copy()
    # A has a robust >=0.5 pIC50 SAR contrast; B is fully unobserved at decision
    # time. B is used here only to assign the outcome, never cohort eligibility.
    b_retained=(data.gap_A*data.gap_B>0)&(data.robust_gap_B>=.3-1e-10)
    b_reversed=(data.gap_A*data.gap_B<0)&(data.robust_gap_B>=.3-1e-10)
    data["outcome"]=np.select([b_retained,b_reversed],["retained","reversed"],default="attenuated_or_uncertain")
    data["binary_loss"]=(~b_retained).astype("int8")
    data["anchor_gap_abs"]=data.gap_A.abs()
    data["anchor_p_mean"]=(data.p_A_1+data.p_A_2)/2
    assert data.comparison_id.is_unique
    save(data,"anchor_transfer_all.parquet")
    counts=[]
    for seed in SEEDS:
        mapping={x:partition(x,seed) for x in set(data.scaffold_1)|set(data.scaffold_2)}
        p1=data.scaffold_1.map(mapping);p2=data.scaffold_2.map(mapping)
        split=pd.DataFrame({"comparison_id":data.comparison_id,"pair_id":data.pair_id,
                            "partition":np.where(p1.eq(p2),p1,"excluded_cross_partition")})
        tr=data.loc[split.partition.eq("train")]
        c=tr.groupby("pair_id").size()
        ids=pd.concat([tr[["pair_id","identity_1"]].rename(columns={"identity_1":"identity"}),
                       tr[["pair_id","identity_2"]].rename(columns={"identity_2":"identity"})]).groupby("pair_id").identity.nunique()
        known=set(c.index[(c>=5)&(ids.reindex(c.index,fill_value=0)>=5)])
        split.loc[split.partition.isin(PARTS)&~split.pair_id.isin(known),"partition"]="excluded_insufficient_train_support"
        save(split,f"anchor_split_scaffold_seed_{seed}.parquet")
        groups={}
        for part in PARTS:
            frame=data.loc[split.partition.eq(part)]
            groups[part]={kind:set(frame[f"{kind}_1"])|set(frame[f"{kind}_2"]) for kind in ["identity","family","scaffold"]}
            counts.append({"seed":seed,"partition":part,"comparisons":len(frame),
                           "loss":int(frame.binary_loss.sum()),"retained":int((1-frame.binary_loss).sum()),
                           "reversed":int(frame.outcome.eq("reversed").sum()),
                           "attenuated_or_uncertain":int(frame.outcome.eq("attenuated_or_uncertain").sum()),
                           "assay_pairs":frame.pair_id.nunique(),"targets":frame.target_chembl_id.nunique(),
                           "molecules":pd.concat([frame.identity_1,frame.identity_2]).nunique(),
                           "loss_assay_pairs":frame.loc[frame.binary_loss.eq(1),"pair_id"].nunique()})
            if seed==17:save(frame,f"anchor_scaffold_{part}.parquet")
        for left,right in [("train","validation"),("train","test"),("validation","test")]:
            for kind in groups[left]:assert groups[left][kind].isdisjoint(groups[right][kind]),(seed,left,right,kind)
        print(f"Seed {seed}: {[(x['partition'],x['comparisons'],x['loss']) for x in counts if x['seed']==seed]}",flush=True)
    pd.DataFrame(counts).to_csv(OUT/"anchor_split_counts.csv",index=False)
    (OUT/"anchor_manifest.json").write_text(json.dumps({
        "task":"Given two similar molecules with >=0.5 pIC50 robust separation in assay A, predict whether the same ordering with >=0.3 pIC50 robust separation remains in assay B.",
        "cohort_eligibility":"Only structure, source assay-A measurements and pair support; no assay-B measurements.",
        "outcome":"Loss includes reversal, attenuation or uncertain replicate interval; reversal is also stored separately.",
        "all_comparisons":len(data),"loss":int(data.binary_loss.sum()),"reversed":int(data.outcome.eq("reversed").sum()),
        "source_sha256":digest(OUT/"all_analogue_comparisons.parquet"),"code_sha256":digest(Path(__file__)),
        "seeds":SEEDS,"thresholds":{"tanimoto":.7,"shared_molecules":5,"A_margin_log10":.5,"B_retention_margin_log10":.3}
    },indent=2))


if __name__=="__main__":main()
