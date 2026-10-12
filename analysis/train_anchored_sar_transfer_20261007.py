"""Fit assay-A-anchored SAR retention models on five scaffold-disjoint splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from prepare_sar_transfer_20261007 import OUT,SEEDS,digest
from train_sar_transfer_baseline_20261007 import read,make_molecule_features,assay_features,target_features

RESULTS=OUT/"models_anchored_20261007"
RESULTS.mkdir(exist_ok=True)


def encode(frame,molecules,assays,median,targets):
    strong=np.where(frame.gap_A.to_numpy()>=0,frame.identity_1,frame.identity_2)
    weak=np.where(frame.gap_A.to_numpy()>=0,frame.identity_2,frame.identity_1)
    f1=np.stack([molecules[x][0] for x in strong])
    f2=np.stack([molecules[x][0] for x in weak])
    d1=np.stack([molecules[x][1] for x in strong])
    d2=np.stack([molecules[x][1] for x in weak])
    molecule=np.concatenate([f1-f2,f1*f2,d1-d2,(d1+d2)/2,
                             frame.tanimoto.to_numpy(dtype=np.float32)[:,None],
                             frame.anchor_gap_abs.to_numpy(dtype=np.float32)[:,None],
                             frame.anchor_p_mean.to_numpy(dtype=np.float32)[:,None]],axis=1).astype(np.float32)
    target=np.stack([targets[x] for x in frame.target_chembl_id]).astype(np.float32)
    left=[assays[x] for x in frame.assay_A]
    right=[assays[x] for x in frame.assay_B]
    ta=np.stack([x[0] for x in left]);tb=np.stack([x[0] for x in right])
    ca=np.stack([x[1] for x in left]);cb=np.stack([x[1] for x in right])
    na=np.stack([x[2] for x in left]);nb=np.stack([x[2] for x in right])
    observed=np.concatenate([np.isfinite(na).astype(np.float32),np.isfinite(nb).astype(np.float32)],axis=1)
    na=np.where(np.isfinite(na),na,median)
    nb=np.where(np.isfinite(nb),nb,median)
    assay=np.concatenate([ta,tb,tb-ta,ca,cb,na,nb,nb-na,observed,
                          frame.source_type.eq("same_document").to_numpy(dtype=np.float32)[:,None],
                          frame.same_description.to_numpy(dtype=np.float32)[:,None]],axis=1).astype(np.float32)
    assert len(molecule)==len(target)==len(assay)==len(frame)
    return {"molecule":molecule,"target":target,"assay":assay}


def metrics(frame,pred):
    y=frame.binary_loss.to_numpy(dtype=int)
    out={"n":len(y),"positives":int(y.sum()),"prevalence":float(y.mean()),
         "AP":float(average_precision_score(y,pred)),
         "AUROC":float(roc_auc_score(y,pred)) if len(set(y))==2 else None,
         "Brier":float(brier_score_loss(y,pred))}
    groups=[]
    for _,idx in frame.groupby("pair_id").indices.items():
        yy=y[idx]
        if len(yy)>=10 and yy.sum()>=3 and len(yy)-yy.sum()>=3:
            groups.append((roc_auc_score(yy,pred[idx]),average_precision_score(yy,pred[idx])))
    out["pair_macro_n"]=len(groups)
    out["pair_macro_AUROC"]=float(np.mean([x[0] for x in groups])) if groups else None
    out["pair_macro_AP"]=float(np.mean([x[1] for x in groups])) if groups else None
    return out


def main(seeds):
    data=read("anchor_transfer_all.parquet")
    ids=set(data.identity_1)|set(data.identity_2)
    molecules=make_molecule_features(ids)
    results=[]
    for seed in seeds:
        split=read(f"anchor_split_scaffold_seed_{seed}.parquet")
        parts={part:data.merge(split.loc[split.partition.eq(part),["comparison_id"]],
                               on="comparison_id",validate="one_to_one") for part in ["train","validation","test"]}
        tr=parts["train"]
        tr_sc=set(tr.scaffold_1)|set(tr.scaffold_2)
        for part in ["validation","test"]:
            assert tr_sc.isdisjoint(set(parts[part].scaffold_1)|set(parts[part].scaffold_2))
        print(f"Seed {seed}: train {len(tr)}, val {len(parts['validation'])}, test {len(parts['test'])}",flush=True)
        context=pd.concat(list(parts.values()),ignore_index=True)
        assays,median,tfidf,svd,formats=assay_features(tr,context)
        targets,pca=target_features(tr,context)
        x={part:encode(frame,molecules,assays,median,targets) for part,frame in parts.items()}
        y=tr.binary_loss.to_numpy(dtype=int)
        prior=float(y.mean())
        g=tr.groupby("pair_id").binary_loss.agg(["sum","count"])
        def historical_prior(frame):
            successes=frame.pair_id.map(g["sum"]).fillna(0).to_numpy()
            n=frame.pair_id.map(g["count"]).fillna(0).to_numpy()
            return (successes+20*prior)/(n+20)
        for part in ["validation","test"]:
            frame=parts[part];pred=historical_prior(frame)
            for cohort,mask in [("all",np.ones(len(frame),dtype=bool)),
                                ("same_document_distinct_description",(frame.source_type.eq("same_document")&~frame.same_description).to_numpy()),
                                ("controlled_condition",frame.controlled_condition_pair.to_numpy())]:
                if mask.sum()>=10:
                    results.append({"seed":seed,"partition":part,"cohort":cohort,
                                    "model":"pair_prior",**metrics(frame.loc[mask].reset_index(drop=True),pred[mask])})
        blocks={"molecule_anchor":["molecule"],"assay_target":["assay","target"],
                "molecule_assay":["molecule","assay"],
                "molecule_target":["molecule","target"],
                "full":["molecule","assay","target"]}
        for name,keys in blocks.items():
            model=ExtraTreesClassifier(n_estimators=128,max_depth=20,min_samples_leaf=4,
                                       max_features="sqrt",class_weight="balanced_subsample",
                                       n_jobs=8,random_state=seed)
            model.fit(np.concatenate([x["train"][k] for k in keys],axis=1),y)
            for part in ["validation","test"]:
                frame=parts[part]
                pred=model.predict_proba(np.concatenate([x[part][k] for k in keys],axis=1))[:,1]
                for cohort,mask in [("all",np.ones(len(frame),dtype=bool)),
                                    ("same_document_distinct_description",(frame.source_type.eq("same_document")&~frame.same_description).to_numpy()),
                                    ("controlled_condition",frame.controlled_condition_pair.to_numpy())]:
                    if mask.sum()>=10:
                        results.append({"seed":seed,"partition":part,"cohort":cohort,
                                        "model":name,**metrics(frame.loc[mask].reset_index(drop=True),pred[mask])})
                if part=="test":
                    o=frame[["comparison_id","pair_id","target_chembl_id","binary_loss","outcome",
                             "source_type","same_description","controlled_condition_pair"]].copy()
                    o["prediction"]=pred
                    o.to_csv(RESULTS/f"predictions_{name}_seed_{seed}.csv",index=False)
            print(f"  {name} finished",flush=True)
            if name=="full":joblib.dump(model,RESULTS/f"full_seed_{seed}.joblib",compress=3)
        joblib.dump({"tfidf":tfidf,"assay_svd":svd,"target_pca":pca,"assay_formats":formats,
                     "numeric_median":median},RESULTS/f"preprocessors_seed_{seed}.joblib",compress=3)
        pd.DataFrame(results).to_csv(RESULTS/"metrics.csv",index=False)
        print(pd.DataFrame(results).query("seed==@seed and partition=='test' and cohort=='all'")[
            ["model","n","positives","AP","AUROC","pair_macro_n","pair_macro_AUROC"]].to_string(index=False),flush=True)
    (RESULTS/"manifest.json").write_text(json.dumps({
        "task":"Predict loss of a measured assay-A SAR advantage when transferred to assay B.",
        "source_sha256":digest(OUT/"anchor_transfer_all.parquet"),"esm_sha256":digest(OUT/"target_esm2_t33_pool.pt"),
        "code_sha256":digest(Path(__file__)),"seeds":seeds,
        "model":"ExtraTreesClassifier, 128 trees, max depth 20, min leaf 4; fixed hyperparameters",
        "features":"A measured pIC50 gap/mean, stronger and weaker molecule Morgan fingerprints and descriptors, assay text SVD and parsed condition numbers, BAO, target ESM2 PCA. No B activity as model input.",
        "evaluation":"All train/validation/test rows scaffold-disjoint; known assay pairs; pair macro metrics only with >=10 rows and >=3 outcomes per class."},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    main(ap.parse_args().seeds)
