"""Build traceable analogue-ranking transfer datasets; no model is trained.

Run from any directory with Python, pandas, numpy, RDKit and duckdb.
Input tables are immutable. Primary endpoint: exact IC50, curated upstream.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/datasets/sar_transfer_20261007"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 23, 41, 71, 101]
PARTS = ["train", "validation", "test"]
CON = duckdb.connect()


def read(path: Path) -> pd.DataFrame:
    return CON.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def save(frame: pd.DataFrame, name: str) -> None:
    CON.register("export_frame", frame)
    escaped = str(OUT / name).replace("'", "''")
    CON.execute(f"COPY export_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    CON.unregister("export_frame")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(value) -> set[str]:
    return set(str(value).split(";")) - {"", "nan", "None"}


def partition(key: str, seed: int) -> str:
    fraction = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16) / 16**16
    return "train" if fraction < .70 else "validation" if fraction < .85 else "test"


def normalize(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def describe(frame: pd.DataFrame) -> dict:
    ids = pd.concat([frame.identity_1, frame.identity_2])
    docs = pd.concat([frame.document_A, frame.document_B])
    return {
        "comparisons": len(frame), "retained": int(frame.label.eq("retained").sum()),
        "reversed": int(frame.label.eq("reversed").sum()),
        "unique_molecules": int(ids.nunique()), "assay_pairs": int(frame.pair_id.nunique()),
        "targets": int(frame.target_chembl_id.nunique()), "documents": int(docs.nunique()),
        "unique_molecular_pairs": int(frame.molecular_pair_id.nunique()),
        "largest_assay_pair_fraction": float(frame.pair_id.value_counts(normalize=True).max()) if len(frame) else None,
    }


def main() -> None:
    sources = {"pairs": ROOT / "data/processed/pairs.parquet",
               "measurements": ROOT / "data/processed/measurements.parquet",
               "molecules": ROOT / "data/processed/molecules.csv",
               "assays": ROOT / "data/processed/assays.csv"}
    p = read(sources["pairs"])
    assert p.endpoint.eq("IC50").all()
    assert not p.duplicated(["pair_id", "chemical_identity"]).any()
    audit = {"raw_molecule_assay_pair_rows": len(p), "raw_assay_pairs": int(p.pair_id.nunique())}
    support_raw = p.groupby("pair_id").size()
    audit["raw_assay_pair_support"] = {str(k): int((support_raw >= k).sum()) for k in [2, 5, 10, 20, 30]}

    # The two assays must have distinct underlying source observations.
    shared = np.array([bool(tokens(a) & tokens(b)) for a, b in zip(
        p.provenance_equivalence_groups_A, p.provenance_equivalence_groups_B)])
    audit["excluded_shared_provenance_rows"] = int(shared.sum())
    p = p.loc[~shared & np.isfinite(p.p_A) & np.isfinite(p.p_B)].copy()
    support = p.groupby("pair_id").chemical_identity.nunique()
    p = p[p.pair_id.isin(support[support >= 2].index)].copy()
    p["shared_molecules"] = p.pair_id.map(support)

    mol = pd.read_csv(sources["molecules"])
    mol = mol[mol.chemical_identity.isin(p.chemical_identity)].drop_duplicates("chemical_identity")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps, invalid = {}, []
    for row in mol.itertuples(index=False):
        structure = Chem.MolFromSmiles(row.canonical_isomeric_smiles)
        if structure is None:
            invalid.append(row.chemical_identity)
        else:
            fps[row.chemical_identity] = gen.GetFingerprint(structure)
    audit["invalid_structure_identities"] = invalid
    assert not invalid, "Unexpected invalid curated structures"
    assert set(p.chemical_identity) <= set(fps)
    p = p.sort_values(["pair_id", "chemical_identity"]).reset_index(drop=True)

    # No scaffold is allowed to split a stereo-stripped identity family.
    assert p.groupby("chemical_family").scaffold.nunique().max() == 1
    m = CON.execute("SELECT measurement_id, p_activity_min, p_activity_max, variant_signal "
                    "FROM read_parquet(?)", [str(sources["measurements"])]).df().set_index("measurement_id")
    for side in ["A", "B"]:
        p[f"min_{side}"] = p[f"measurement_id_{side}"].map(m.p_activity_min)
        p[f"max_{side}"] = p[f"measurement_id_{side}"].map(m.p_activity_max)
        p[f"variant_{side}"] = p[f"measurement_id_{side}"].map(m.variant_signal)
    assert p[["min_A", "max_A", "min_B", "max_B"]].notna().all().all()

    assay = pd.read_csv(sources["assays"])
    assay = assay[assay.standard_type.eq("IC50")].drop_duplicates("assay_chembl_id")
    assay["normalized_description"] = assay.assay_description.fillna("").map(normalize)
    descriptions = assay.set_index("assay_chembl_id").normalized_description
    p["same_description"] = p.assay_A.map(descriptions).eq(p.assay_B.map(descriptions))
    controlled_path = ROOT / "outputs/audit/controlled_condition_scan_20261007/observed_controlled_condition_pairs.csv"
    controlled = pd.read_csv(controlled_path)
    controlled = controlled[controlled.endpoint.eq("IC50") & ~controlled.feature.eq("construct")]
    controlled_keys = {(t, *sorted([a,b])) for t,a,b in zip(controlled.target, controlled.assay_A, controlled.assay_B)}
    p["controlled_condition_pair"] = [(t, *sorted([a,b])) in controlled_keys
                                      for t,a,b in zip(p.target_chembl_id,p.assay_A,p.assay_B)]

    first, second, similarity = [], [], []
    total_combinations = 0
    for count, (_, g) in enumerate(p.groupby("pair_id", sort=True)):
        idx = g.index.to_numpy()
        vectors = [fps[x] for x in g.chemical_identity]
        total_combinations += len(idx) * (len(idx) - 1) // 2
        for i in range(len(idx)-1):
            sim = np.asarray(DataStructs.BulkTanimotoSimilarity(vectors[i], vectors[i+1:]))
            offsets = np.flatnonzero(sim >= .60)
            first.extend([idx[i]] * len(offsets))
            second.extend(idx[i+1+offsets].tolist())
            similarity.extend(sim[offsets].tolist())
        if count % 2000 == 0:
            print(f"Enumerated {count} assay pairs; {len(first)} analogue comparisons", flush=True)

    a = p.iloc[first].reset_index(drop=True)
    b = p.iloc[second].reset_index(drop=True)
    keep_columns = ["pair_id", "target_chembl_id", "assay_A", "assay_B", "document_A", "document_B",
                    "source_type", "shared_molecules", "same_description", "controlled_condition_pair"]
    rows = a[keep_columns].copy()
    for index, frame in [(1, a), (2, b)]:
        for old, new in [("chemical_identity", "identity"), ("chemical_family", "family"),
                         ("scaffold", "scaffold"), ("sample_id", "sample_id")]:
            rows[f"{new}_{index}"] = frame[old]
        for side in ["A", "B"]:
            for column in ["p", "min", "max", "measurement_id", "activity_ids", "provenance_equivalence_groups"]:
                rows[f"{column}_{side}_{index}"] = frame[f"{column}_{side}"]
    rows["tanimoto"] = similarity
    rows["same_scaffold"] = rows.scaffold_1.eq(rows.scaffold_2)
    rows["variant_flag"] = (a.variant_A | a.variant_B | b.variant_A | b.variant_B).astype(bool)
    rows["molecular_pair_id"] = [hashlib.sha256(f"{x}|{y}".encode()).hexdigest()[:24]
                                 for x, y in zip(rows.identity_1, rows.identity_2)]
    rows["comparison_id"] = ["sar_"+hashlib.sha256(f"{x}|{y}".encode()).hexdigest()[:24]
                             for x, y in zip(rows.pair_id, rows.molecular_pair_id)]
    for side in ["A", "B"]:
        rows[f"gap_{side}"] = rows[f"p_{side}_1"] - rows[f"p_{side}_2"]
        rows[f"robust_gap_{side}"] = np.where(
            rows[f"gap_{side}"] >= 0,
            rows[f"min_{side}_1"] - rows[f"max_{side}_2"],
            rows[f"min_{side}_2"] - rows[f"max_{side}_1"],
        )
    rows["sar_change"] = rows.gap_B - rows.gap_A
    strong_a = rows.robust_gap_A >= .5 - 1e-10
    strong_b = rows.robust_gap_B >= .5 - 1e-10
    rows["label"] = np.select(
        [strong_a & strong_b & (rows.gap_A * rows.gap_B > 0),
         strong_a & strong_b & (rows.gap_A * rows.gap_B < 0),
         strong_a & ~strong_b, ~strong_a & strong_b],
        ["retained", "reversed", "A_only_resolved", "B_only_resolved"], default="unresolved")
    rows["binary_reversal"] = rows.label.map({"retained": 0, "reversed": 1}).astype("Int64")
    rows["primary_eligible"] = (rows.tanimoto.ge(.7) & rows.shared_molecules.ge(10)
                                & rows.binary_reversal.notna())
    rows["same_document_distinct_protocol"] = rows.source_type.eq("same_document") & ~rows.same_description
    assert not rows.comparison_id.duplicated().any()
    assert rows.identity_1.lt(rows.identity_2).all()
    save(rows, "all_analogue_comparisons.parquet")
    save(p, "molecule_assay_observations.parquet")
    save(mol, "molecules.parquet")
    save(assay[assay.assay_chembl_id.isin(set(p.assay_A)|set(p.assay_B))], "assay_contexts.parquet")
    audit.update({"candidate_four_value_comparisons": total_combinations,
                  "comparisons_tanimoto_ge_0_6": len(rows),
                  "labels_tanimoto_ge_0_7": rows.loc[rows.tanimoto.ge(.7), "label"].value_counts().to_dict()})

    # Count prespecified thresholds, without choosing a threshold from class balance.
    sensitivity = []
    for tc, margin, minimum in itertools.product([.6, .7, .8], [.3, .5, 1.0], [2, 5, 10, 20, 30]):
        subset = rows[(rows.tanimoto >= tc) & (rows.shared_molecules >= minimum)
                      & (rows.robust_gap_A >= margin-1e-10) & (rows.robust_gap_B >= margin-1e-10)]
        sensitivity.append({"tanimoto_min": tc, "gap_min_log10": margin, "min_shared_molecules": minimum,
                            "comparisons": len(subset), "reversed": int((subset.gap_A*subset.gap_B < 0).sum()),
                            "assay_pairs": int(subset.pair_id.nunique()), "targets": int(subset.target_chembl_id.nunique())})
    pd.DataFrame(sensitivity).to_csv(OUT / "threshold_sensitivity.csv", index=False)

    primary = rows[rows.primary_eligible].copy()
    save(primary, "primary_comparisons.parquet")
    coverage = []
    for name, mask in {
        "all_sources": pd.Series(True, index=primary.index),
        "same_document": primary.source_type.eq("same_document"),
        "cross_document": primary.source_type.eq("cross_document"),
        "same_document_distinct_protocol": primary.same_document_distinct_protocol,
        "controlled_condition_pairs": primary.controlled_condition_pair,
        "variant_flag_absent": ~primary.variant_flag,
    }.items():
        coverage.append({"cohort": name, **describe(primary[mask])})
    pd.DataFrame(coverage).to_csv(OUT / "cohort_coverage.csv", index=False)

    pair_inventory = primary.groupby(["pair_id", "target_chembl_id", "assay_A", "assay_B", "source_type", "document_A", "document_B"]).agg(
        comparisons=("comparison_id", "size"), reversed=("binary_reversal", "sum"),
        shared_molecules=("shared_molecules", "first"), same_description=("same_description", "first"),
        controlled_condition_pair=("controlled_condition_pair", "first")).reset_index()
    pair_inventory["retained"] = pair_inventory.comparisons - pair_inventory.reversed
    pair_inventory.to_csv(OUT / "assay_pair_inventory.csv", index=False)

    summaries, overlap_audits, allocations, reasons = [], [], [], []
    for scheme, group_column in [("family_disjoint", "family"), ("scaffold_disjoint", "scaffold")]:
        for seed in SEEDS:
            split = primary[["comparison_id", "pair_id", "binary_reversal"]].copy()
            keys = set(primary[f"{group_column}_1"]) | set(primary[f"{group_column}_2"])
            mapping = {x: partition(x, seed) for x in keys}
            part1, part2 = primary[f"{group_column}_1"].map(mapping), primary[f"{group_column}_2"].map(mapping)
            split["partition"] = np.where(part1.eq(part2), part1, "excluded_cross_partition")
            # Known-pair deployment: an assay pair needs calibration examples from train.
            training = primary.loc[split.partition.eq("train")]
            train_support = training.groupby("pair_id").agg(n_comparisons=("comparison_id", "size"))
            train_ids = pd.concat([training[["pair_id", "identity_1"]].rename(columns={"identity_1":"identity"}),
                                   training[["pair_id", "identity_2"]].rename(columns={"identity_2":"identity"})])
            train_support["n_molecules"] = train_ids.groupby("pair_id").identity.nunique()
            known = train_support.index[(train_support.n_comparisons >= 5) & (train_support.n_molecules >= 5)]
            split.loc[split.partition.isin(PARTS) & ~split.pair_id.isin(known), "partition"] = "excluded_insufficient_train_support"
            split["pair_balanced_weight"] = np.nan
            for part in PARTS:
                mask = split.partition.eq(part)
                counts = split.loc[mask].groupby("pair_id").comparison_id.transform("size")
                split.loc[mask, "pair_balanced_weight"] = 1 / counts
                subset = primary.loc[mask]
                summaries.append({"scheme": scheme, "seed": seed, "partition": part, **describe(subset)})
                for cohort, cmask in [("same_document_distinct_protocol", subset.same_document_distinct_protocol),
                                     ("controlled_condition_pairs", subset.controlled_condition_pair)]:
                    summaries.append({"scheme": scheme+"__"+cohort, "seed": seed, "partition": part, **describe(subset[cmask])})

            # Audit every split against identities, stereo families, raw measurements,
            # activity identifiers and provenance groups, not merely comparison IDs.
            sets = {}
            for part in PARTS:
                sub = primary.loc[split.partition.eq(part)]
                sets[part] = {}
                for kind in ["identity", "family", "scaffold", "measurement_id", "activity_ids", "provenance_equivalence_groups"]:
                    columns = [f"{kind}_{i}" for i in [1,2]] if kind in ["identity","family","scaffold"] else [f"{kind}_{s}_{i}" for s in ["A","B"] for i in [1,2]]
                    sets[part][kind] = set().union(*(tokens(v) for c in columns for v in sub[c]))
                sets[part]["assay"] = set(sub.assay_A) | set(sub.assay_B)
                sets[part]["document"] = set(sub.document_A) | set(sub.document_B)
                sets[part]["assay_pair"] = set(sub.pair_id)
            for left, right in itertools.combinations(PARTS,2):
                for kind in sets[left]:
                    overlap = len(sets[left][kind] & sets[right][kind])
                    expected_disjoint = kind in {"identity","family","measurement_id","activity_ids","provenance_equivalence_groups"} or (scheme=="scaffold_disjoint" and kind=="scaffold")
                    overlap_audits.append({"scheme":scheme,"seed":seed,"left":left,"right":right,
                                           "entity":kind,"overlap":overlap,"required_zero":expected_disjoint})
                    if expected_disjoint:
                        assert overlap == 0, (scheme, seed, left, right, kind, overlap)
            for status, count in split.partition.value_counts().items():
                reasons.append({"scheme":scheme,"seed":seed,"status":status,"comparisons":int(count)})
            save(split, f"split_{scheme}_seed_{seed}.parquet")
            if seed == 17:
                for part in PARTS:
                    indices = split.partition.eq(part)
                    export = primary.loc[indices].copy()
                    export["pair_balanced_weight"] = split.loc[indices,"pair_balanced_weight"]
                    save(export, f"{scheme}_{part}.parquet")
            for key, part in mapping.items():
                allocations.append({"scheme":scheme,"seed":seed,"group":key,"partition":part})
            print(f"Finished {scheme}, seed {seed}: {split.partition.value_counts().to_dict()}", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "split_counts.csv", index=False)
    pd.DataFrame(overlap_audits).to_csv(OUT / "split_overlap_audit.csv", index=False)
    pd.DataFrame(reasons).to_csv(OUT / "split_exclusions.csv", index=False)
    save(pd.DataFrame(allocations), "group_assignments.parquet")
    audit["primary_dataset"] = describe(primary)
    audit["versions"] = {"rdkit": rdBase.rdkitVersion, "pandas": pd.__version__, "duckdb": duckdb.__version__}
    audit["parameters"] = {"primary_tanimoto":.7,"morgan_radius":2,"morgan_bits":2048,"chirality":True,
                            "margin_log10_each_assay":.5,"minimum_shared_molecules":10,
                            "known_pair_min_training_molecules":5,"known_pair_min_training_comparisons":5,
                            "group_split_ratios":[.70,.15,.15],"seeds":SEEDS,
                            "replicate_rule":"Worst-case min/max observed replicate gaps must pass the margin; these are not confidence intervals"}
    audit["source_sha256"] = {name:digest(path) for name,path in sources.items()}
    audit["source_sha256"]["controlled_condition_pairs"] = digest(controlled_path)
    audit["code_sha256"] = digest(Path(__file__))
    (OUT / "manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit["primary_dataset"],indent=2), flush=True)


if __name__ == "__main__":
    main()
