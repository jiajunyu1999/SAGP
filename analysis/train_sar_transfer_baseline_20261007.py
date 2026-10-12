"""Simple scaffold-held-out SAR reversal baseline with ESM, assay and chemistry.

Uses label-blind ChEMBL37 assay text/numeric metadata, frozen ESM2 target
embeddings, and Morgan/physicochemical molecule features. Predicts strict IC50
reversal labels. The expanded IC50 train rows are softer labels; strict held-out
validation/test partitions are frozen independently of the expansion.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from prepare_sar_transfer_20261007 import OUT, SEEDS, PARTS, save, digest

RESULTS=OUT/"models_simple_20261007"
RESULTS.mkdir(exist_ok=True)
CON=duckdb.connect()
FP_BITS=512


def read(name):
    return CON.execute("SELECT * FROM read_parquet(?)",[str(OUT/name)]).df()


def make_molecule_features(ids):
    mol=read("molecules.parquet").drop_duplicates("chemical_identity")
    mol=mol[mol.chemical_identity.isin(ids)]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=FP_BITS,includeChirality=True)
    vectors={}
    for row in mol.itertuples(index=False):
        structure=Chem.MolFromSmiles(row.canonical_isomeric_smiles)
        assert structure is not None
        bits=np.zeros(FP_BITS,dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(structure),bits)
        props=np.array([
            Descriptors.MolWt(structure),Descriptors.MolLogP(structure),
            rdMolDescriptors.CalcTPSA(structure),rdMolDescriptors.CalcNumHBD(structure),
            rdMolDescriptors.CalcNumHBA(structure),rdMolDescriptors.CalcNumRotatableBonds(structure),
            rdMolDescriptors.CalcFractionCSP3(structure),rdMolDescriptors.CalcNumAromaticRings(structure),
            rdMolDescriptors.CalcNumRings(structure),structure.GetNumHeavyAtoms(),
            Chem.GetFormalCharge(structure),
        ],dtype=np.float32)
        vectors[row.chemical_identity]=(bits,props)
    assert set(ids)<=set(vectors)
    return vectors


def numeric_assay(text):
    text=str(text)
    fields=[]
    units={"pm":1e-3,"nm":1,"um":1e3,"µm":1e3,"μm":1e3,"mm":1e6,"m":1e9}
    for name in ["atp","nad","nadh","nadph","dtt","bsa"]:
        pattern=rf"\b{name}\b[^.;]{{0,35}}?(\d+(?:\.\d+)?)\s*(pM|nM|uM|µM|μM|mM|M)\b"
        match=re.search(pattern,text,re.I)
        if match is None:
            pattern=rf"(\d+(?:\.\d+)?)\s*(pM|nM|uM|µM|μM|mM|M)\s*(?:of\s+)?{name}\b"
            match=re.search(pattern,text,re.I)
        value=float(match.group(1))*units[match.group(2).lower()] if match else np.nan
        fields.append(np.log10(value) if value>0 else np.nan)
    ph=re.search(r"\bpH\s*(?:of\s*|=\s*|:\s*)?(\d+(?:\.\d+)?)",text,re.I)
    fields.append(float(ph.group(1)) if ph else np.nan)
    duration=re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|h|min|s)\b",text,re.I)
    if duration:
        unit=duration.group(2).lower()
        mult=60 if unit.startswith("h") else 1/60 if unit.startswith("s") else 1
        fields.append(np.log1p(float(duration.group(1))*mult))
    else:fields.append(np.nan)
    temp=re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*|degrees?\s*|deg\.?\s*)C\b",text,re.I)
    fields.append(float(temp.group(1)) if temp else np.nan)
    return np.asarray(fields,dtype=np.float32)


def assay_features(training,all_rows):
    assay=read("assay_contexts.parquet").drop_duplicates("assay_chembl_id").set_index("assay_chembl_id")
    train_ids=set(training.assay_A)|set(training.assay_B)
    needed=set(all_rows.assay_A)|set(all_rows.assay_B)
    assert needed<=set(assay.index)
    texts=assay.assay_description.fillna("").astype(str)
    vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=5000,min_df=2,sublinear_tf=True,
                               stop_words="english",dtype=np.float32)
    train_text=vectorizer.fit_transform(texts.loc[sorted(train_ids)])
    rank=min(24,train_text.shape[0]-1,train_text.shape[1]-1)
    svd=TruncatedSVD(n_components=rank,random_state=17).fit(train_text)
    textvec=svd.transform(vectorizer.transform(texts.loc[sorted(needed)])).astype(np.float32)
    cat=assay.bao_format.fillna("missing").astype(str)
    formats=sorted(cat.loc[sorted(train_ids)].unique())
    numeric={x:numeric_assay(texts.loc[x]) for x in needed}
    nums=np.stack([numeric[x] for x in sorted(train_ids)])
    median=np.asarray([np.nanmedian(nums[:,j]) if np.isfinite(nums[:,j]).any() else 0.0
                       for j in range(nums.shape[1])],dtype=np.float32)
    catvec={x:np.asarray([cat.loc[x]==y for y in formats],dtype=np.float32) for x in needed}
    return {x:(textvec[i],catvec[x],numeric[x]) for i,x in enumerate(sorted(needed))},median,vectorizer,svd,formats


def target_features(training,all_rows):
    embeddings=torch.load(OUT/"target_esm2_t33_pool.pt",map_location="cpu",weights_only=True)
    needed=set(all_rows.target_chembl_id)
    assert needed<=set(embeddings)
    train_ids=sorted(set(training.target_chembl_id))
    pca=PCA(n_components=min(32,len(train_ids)-1),svd_solver="randomized",random_state=17)
    pca.fit(np.stack([embeddings[x].numpy() for x in train_ids]))
    vectors={x:pca.transform(embeddings[x].numpy().reshape(1,-1))[0].astype(np.float32) for x in needed}
    return vectors,pca


def encode(rows,molecules,assays,numeric_median,targets):
    n=len(rows)
    first=np.stack([molecules[x][0] for x in rows.identity_1])
    second=np.stack([molecules[x][0] for x in rows.identity_2])
    pfirst=np.stack([molecules[x][1] for x in rows.identity_1])
    psecond=np.stack([molecules[x][1] for x in rows.identity_2])
    molblocks=[np.abs(first-second),first*second,(pfirst+psecond)/2,np.abs(pfirst-psecond),
               rows.tanimoto.to_numpy(dtype=np.float32).reshape(-1,1)]
    mol=np.concatenate(molblocks,axis=1).astype(np.float32)
    target=np.stack([targets[x] for x in rows.target_chembl_id]).astype(np.float32)
    asa=[assays[x] for x in rows.assay_A]
    asb=[assays[x] for x in rows.assay_B]
    ta=np.stack([x[0] for x in asa]);tb=np.stack([x[0] for x in asb])
    ca=np.stack([x[1] for x in asa]);cb=np.stack([x[1] for x in asb])
    na=np.stack([x[2] for x in asa]);nb=np.stack([x[2] for x in asb])
    observed=np.isfinite(na).astype(np.float32)+np.isfinite(nb).astype(np.float32)
    na=np.where(np.isfinite(na),na,numeric_median)
    nb=np.where(np.isfinite(nb),nb,numeric_median)
    assay=np.concatenate([(ta+tb)/2,np.abs(ta-tb),(ca+cb)/2,np.abs(ca-cb),
                          (na+nb)/2,np.abs(na-nb),observed,
                          rows.source_type.eq("same_document").to_numpy(dtype=np.float32).reshape(-1,1),
                          rows.same_description.to_numpy(dtype=np.float32).reshape(-1,1)],axis=1)
    assert mol.shape[0]==target.shape[0]==assay.shape[0]==n
    return mol,target,assay.astype(np.float32)


def scores(rows,pred):
    y=rows.binary_reversal.to_numpy(dtype=int)
    out={"n":len(rows),"positive":int(y.sum()),"prevalence":float(y.mean()),
         "AP":float(average_precision_score(y,pred)),
         "AUROC":float(roc_auc_score(y,pred)) if len(set(y))==2 else None,
         "Brier":float(brier_score_loss(y,pred))}
    groups=[]
    for _,idx in rows.groupby("pair_id").indices.items():
        yy=y[idx]; pp=pred[idx]
        if len(yy)>=10 and yy.sum()>=3 and (len(yy)-yy.sum())>=3:
            groups.append((roc_auc_score(yy,pp),average_precision_score(yy,pp)))
    out["pair_macro_n"]=len(groups)
    out["pair_macro_AUROC"]=float(np.mean([x[0] for x in groups])) if groups else None
    out["pair_macro_AP"]=float(np.mean([x[1] for x in groups])) if groups else None
    return out


def main(seeds):
    expanded=read("expanded_comparisons.parquet")
    strict=read("primary_comparisons.parquet")
    universe=pd.concat([expanded[["identity_1","identity_2"]],strict[["identity_1","identity_2"]]])
    molecule_ids=set(universe.identity_1)|set(universe.identity_2)
    molecules=make_molecule_features(molecule_ids)
    results=[]
    for seed in seeds:
        esplit=read(f"expanded_split_scaffold_seed_{seed}.parquet")
        ssplit=read(f"split_scaffold_disjoint_seed_{seed}.parquet")
        train=expanded.merge(esplit.loc[esplit.partition.eq("train"),["comparison_id"]],on="comparison_id",validate="one_to_one")
        sets={name:strict.merge(ssplit.loc[ssplit.partition.eq(name),["comparison_id"]],on="comparison_id",validate="one_to_one")
              for name in ["validation","test"]}
        all_rows=pd.concat([train,*sets.values()],ignore_index=True)
        tr_sc=set(train.scaffold_1)|set(train.scaffold_2)
        for part,frame in sets.items():
            assert tr_sc.isdisjoint(set(frame.scaffold_1)|set(frame.scaffold_2)),(seed,part)
        print(f"Seed {seed} train {len(train)} validation {len(sets['validation'])} test {len(sets['test'])}",flush=True)
        assays,med,vect,svd,formats=assay_features(train,all_rows)
        targets,pca=target_features(train,all_rows)
        chunks={}
        for name,frame in [("train",train),*sets.items()]:
            blocks=encode(frame,molecules,assays,med,targets)
            chunks[name]={"molecule":blocks[0],"target":blocks[1],"assay":blocks[2]}
        y=train.binary_reversal.to_numpy(dtype=int)
        # Fixed hyperparameters; no held-out tuning. Compare data expansion
        # against an assay-pair history baseline and feature ablations.
        prior=float(y.mean())
        counts=train.groupby("pair_id").binary_reversal.agg(["sum","count"])
        def pair_prior(frame):
            s=frame.pair_id.map(counts["sum"]).fillna(0).to_numpy()
            n=frame.pair_id.map(counts["count"]).fillna(0).to_numpy()
            return (s+20*prior)/(n+20)
        for name,frame in sets.items():
            pred=pair_prior(frame)
            results.append({"seed":seed,"partition":name,"model":"pair_prior",**scores(frame,pred)})

        kinds={"molecule_only":["molecule"],"assay_target":["assay","target"],
               "full":["molecule","target","assay"]}
        for kind,blocks in kinds.items():
            xtr=np.concatenate([chunks["train"][key] for key in blocks],axis=1)
            model=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=4,max_depth=20,
                                       max_features="sqrt",class_weight="balanced_subsample",
                                       n_jobs=8,random_state=seed)
            model.fit(xtr,y)
            for name,frame in sets.items():
                x=np.concatenate([chunks[name][key] for key in blocks],axis=1)
                pred=model.predict_proba(x)[:,1]
                results.append({"seed":seed,"partition":name,"model":kind,**scores(frame,pred)})
                if name=="test":
                    out=frame[["comparison_id","pair_id","target_chembl_id","binary_reversal","source_type","same_description","controlled_condition_pair"]].copy()
                    out["prediction"]=pred
                    out.to_csv(RESULTS/f"predictions_{kind}_seed_{seed}.csv",index=False)
            print(f"  {kind} complete",flush=True)
            if kind=="full":
                joblib.dump(model,RESULTS/f"full_seed_{seed}.joblib",compress=3)
        joblib.dump({"tfidf":vect,"svd":svd,"target_pca":pca,"assay_formats":formats,"numeric_median":med},
                    RESULTS/f"preprocessors_seed_{seed}.joblib",compress=3)
        pd.DataFrame(results).to_csv(RESULTS/"metrics.csv",index=False)
        print(pd.DataFrame(results).query("seed==@seed and partition=='test'")[
            ["model","n","positive","AP","AUROC","pair_macro_n","pair_macro_AUROC"]].to_string(index=False),flush=True)
    manifest={"source_expanded_sha256":digest(OUT/"expanded_comparisons.parquet"),
              "source_strict_sha256":digest(OUT/"primary_comparisons.parquet"),
              "esm_sha256":digest(OUT/"target_esm2_t33_pool.pt"),
              "code_sha256":digest(Path(__file__)),"seeds":seeds,"morgan_bits":FP_BITS,
              "models":"ExtraTreesClassifier, 128 trees, depth 20, leaf 4, balanced_subsample; pair prior beta-like smoothing strength 20",
              "input_policy":"Structure, assay description/BAO/numeric mentions and target ESM only; no measured pIC50 in features. Fitted text SVD/PCA on train only. Test strict frozen benchmark, train expanded softer label benchmark."}
    (RESULTS/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    args=parser.parse_args()
    main(args.seeds)
