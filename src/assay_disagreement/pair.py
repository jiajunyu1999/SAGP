"""Aggregate primitive measurements and emit traceable assay pairs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dump_json, ensure_dirs, load_config, project_root, sha256, write_table


def stable_id(prefix: str, *parts: object) -> str:
    return prefix + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def aggregate_measurements(curated: pd.DataFrame, release: int) -> pd.DataFrame:
    keys = ["chemical_identity", "chemical_family", "scaffold", "target_chembl_id", "standard_type",
            "assay_id", "assay_chembl_id", "assay_doc_id", "variant_signal"]
    # Most exact ChEMBL records occur once for a molecule/target/assay key.
    # Build these singleton groups without Python ``groupby`` iteration and
    # reserve the more expensive provenance-preserving aggregation for genuine
    # replicate groups. This preserves the output schema and median rule while
    # making a complete-release rebuild tractable.
    group_size = curated.groupby(keys, dropna=False, sort=False)["activity_id"].transform("size")
    single = curated.loc[group_size.eq(1)].copy()
    multi = curated.loc[group_size.gt(1)].copy()

    single_out = pd.DataFrame({
        "release": release,
        **{key: single[key].to_numpy() for key in keys},
        "molecule_chembl_id": single.molecule_chembl_id.astype(str).to_numpy(),
        "canonical_isomeric_smiles": single.canonical_isomeric_smiles.to_numpy(),
        "assay_description": single.assay_description.to_numpy(),
        "assay_type": single.assay_type.to_numpy(),
        "bao_format": single.bao_format.to_numpy(),
        "target_name": single.target_name.to_numpy(),
        "document_year": single.document_year.to_numpy(),
        "document_doi": single.document_doi.fillna("").to_numpy(),
        "p_activity": single.p_activity.astype(float).to_numpy(),
        "p_activity_min": single.p_activity.astype(float).to_numpy(),
        "p_activity_max": single.p_activity.astype(float).to_numpy(),
        "p_activity_iqr": np.zeros(len(single), dtype=float),
        "n_records": np.ones(len(single), dtype=np.int64),
        "activity_ids": single.activity_id.astype(str).to_numpy(),
        "provenance_equivalence_groups": single.provenance_equivalence_group.astype(str).to_numpy(),
    })
    single_out.insert(0, "measurement_id", [
        stable_id("meas_", release, *key)
        for key in single[keys].itertuples(index=False, name=None)
    ])

    rows = []
    for key, group in multi.groupby(keys, dropna=False, sort=True):
        values = group["p_activity"].astype(float)
        rows.append({
            "measurement_id": stable_id("meas_", release, *key),
            "release": release,
            **dict(zip(keys, key)),
            "molecule_chembl_id": ";".join(sorted(group.molecule_chembl_id.astype(str).unique())),
            "canonical_isomeric_smiles": group.canonical_isomeric_smiles.iloc[0],
            "assay_description": group.assay_description.iloc[0],
            "assay_type": group.assay_type.iloc[0],
            "bao_format": group.bao_format.iloc[0],
            "target_name": group.target_name.iloc[0],
            "document_year": group.document_year.iloc[0] if group.document_year.notna().any() else np.nan,
            "document_doi": group.document_doi.iloc[0] if group.document_doi.notna().any() else "",
            "p_activity": float(values.median()),
            "p_activity_min": float(values.min()),
            "p_activity_max": float(values.max()),
            "p_activity_iqr": float(values.quantile(.75) - values.quantile(.25)),
            "n_records": int(len(group)),
            "activity_ids": ";".join(map(str, sorted(group.activity_id.astype(str).unique()))),
            "provenance_equivalence_groups": ";".join(sorted(group.provenance_equivalence_group.astype(str).unique())),
        })
    multi_out = pd.DataFrame(rows, columns=single_out.columns)
    result = pd.concat([single_out, multi_out], ignore_index=True)
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def make_pairs(measurements: pd.DataFrame, release: int) -> pd.DataFrame:
    pair_rows = []
    grouping = ["chemical_identity", "chemical_family", "scaffold", "target_chembl_id", "standard_type", "variant_signal"]
    for _, group in measurements.groupby(grouping, dropna=False, sort=True):
        ordered = group.sort_values("assay_id").to_dict("records")
        for left, right in itertools.combinations(ordered, 2):
            if left["assay_id"] == right["assay_id"]:
                continue
            signed = float(right["p_activity"] - left["p_activity"])
            missing_left_doc = pd.isna(left["assay_doc_id"]) or str(left["assay_doc_id"]) == ""
            missing_right_doc = pd.isna(right["assay_doc_id"]) or str(right["assay_doc_id"]) == ""
            source_type = "unknown" if missing_left_doc or missing_right_doc else (
                "same_document" if left["assay_doc_id"] == right["assay_doc_id"] else "cross_document")
            pair_id = stable_id("pair_", release, left["target_chembl_id"], left["standard_type"],
                                left["assay_chembl_id"], right["assay_chembl_id"])
            pair_rows.append({
                "sample_id": stable_id("sample_", release, left["chemical_identity"], left["target_chembl_id"],
                                        left["standard_type"], pair_id),
                "release": release, "pair_id": pair_id,
                "chemical_identity": left["chemical_identity"], "chemical_family": left["chemical_family"],
                "scaffold": left["scaffold"], "target_chembl_id": left["target_chembl_id"],
                "endpoint": left["standard_type"], "assay_A": left["assay_chembl_id"],
                "assay_B": right["assay_chembl_id"], "assay_id_A": left["assay_id"],
                "assay_id_B": right["assay_id"], "measurement_id_A": left["measurement_id"],
                "measurement_id_B": right["measurement_id"], "activity_ids_A": left["activity_ids"],
                "activity_ids_B": right["activity_ids"], "p_A": left["p_activity"],
                "p_B": right["p_activity"], "signed_delta": signed, "abs_delta": abs(signed),
                "raw_label_ge_1": int(abs(signed) >= 1.0), "source_type": source_type,
                "document_A": left["assay_doc_id"], "document_B": right["assay_doc_id"],
                "provenance_equivalence_groups_A": left.get("provenance_equivalence_groups", ""),
                "provenance_equivalence_groups_B": right.get("provenance_equivalence_groups", ""),
                "n_records_A": left["n_records"], "n_records_B": right["n_records"],
            })
    return pd.DataFrame(pair_rows)


def apply_pilot_cap(pairs: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    pilot = config.get("pilot", {})
    cap = pilot.get("paired_row_cap")
    target_cap = pilot.get("initial_target_cap")
    out = pairs.copy()
    info = {"pilot_target_cap": target_cap, "pilot_pair_cap": cap, "cap_applied": False}
    if target_cap and len(out):
        target_counts = out.groupby("target_chembl_id").chemical_family.nunique().sort_values(ascending=False)
        keep_targets = target_counts.head(int(target_cap)).index
        out = out[out.target_chembl_id.isin(keep_targets)].copy()
        info["selected_targets"] = list(map(str, keep_targets))
    if cap and len(out) > int(cap):
        fam_order = sorted(out.chemical_family.dropna().unique(),
                           key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())
        keep = set()
        for family in fam_order:
            n = int((out.chemical_family == family).sum())
            if keep and sum(int(out.chemical_family.isin(keep))) + n > int(cap):
                continue
            keep.add(family)
        out = out[out.chemical_family.isin(keep)].copy()
        info["cap_applied"] = True
    return out.sort_values("sample_id").reset_index(drop=True), info


def audit_pairs(pairs: pd.DataFrame, measurements: pd.DataFrame, extraction: dict, curation: dict) -> dict:
    extraction = {**extraction}
    extraction.setdefault("raw_activity_rows", extraction.get("activity_rows_returned", 0))
    thresholds = {str(t): float((pairs.abs_delta >= t).mean()) if len(pairs) else None
                  for t in [0.5, 0.7, 1.0, 1.5]}
    bins = [-np.inf, 1, 4, 9, 19, 49, np.inf]
    labels = ["1", "2-4", "5-9", "10-19", "20-49", ">=50"]
    overlap = pairs.groupby(["target_chembl_id", "assay_A", "assay_B"], sort=True).chemical_identity.nunique()
    identity_pair_counts = pairs.groupby(["chemical_identity", "target_chembl_id", "endpoint"], sort=True).size()
    return {
        **extraction, **curation, "measurement_rows": int(len(measurements)), "pair_rows": int(len(pairs)),
        "paired_identity_count": int(pairs.chemical_identity.nunique()) if len(pairs) else 0,
        "assay_pair_count_ge_2_shared": int((overlap >= 2).sum()),
        "shared_identity_bins": pd.cut(overlap, bins=bins, labels=labels).value_counts().sort_index().astype(int).to_dict(),
        "source_type_fraction": pairs.source_type.value_counts(normalize=True).to_dict() if len(pairs) else {},
        "abs_delta_summary": pairs.abs_delta.describe().to_dict() if len(pairs) else {},
        "abs_delta_threshold_fraction": thresholds,
        "offset_coverage_support_ge_10": float((overlap >= 10).mean()) if len(overlap) else 0.0,
        "unique_targets": int(pairs.target_chembl_id.nunique()) if len(pairs) else 0,
        "unique_assays": int(pd.concat([pairs.assay_A, pairs.assay_B]).nunique()) if len(pairs) else 0,
        "unique_documents": int(pd.concat([pairs.document_A, pairs.document_B]).replace("", np.nan).nunique()) if len(pairs) else 0,
        "pair_rows_per_identity_summary": identity_pair_counts.describe().to_dict() if len(identity_pair_counts) else {},
        "pair_row_weight_sum_max_error": float(pairs.groupby(
            ["chemical_identity", "target_chembl_id", "endpoint"], sort=True).pair_row_weight.sum().sub(1.0).abs().max())
        if len(pairs) else 0.0,
    }


def run(config: dict, root: Path) -> None:
    curated = pd.read_csv(root / "data/processed/curated_measurements.csv", low_memory=False)
    measurements = aggregate_measurements(curated, int(config["project"]["release"]))
    full_pairs = make_pairs(measurements, int(config["project"]["release"]))
    pair_counts = full_pairs.groupby(["chemical_identity", "target_chembl_id", "endpoint"], sort=False).size()
    full_pairs["pair_rows_per_identity"] = full_pairs.set_index(
        ["chemical_identity", "target_chembl_id", "endpoint"]
    ).index.map(pair_counts).to_numpy(dtype=np.int64)
    full_pairs["pair_row_weight"] = 1.0 / full_pairs["pair_rows_per_identity"]
    primary_endpoint = str(config["data"].get("endpoint_primary", "IC50"))
    external_endpoint = str(config["data"].get("endpoint_external", "Ki"))
    primary_pairs, cap_info = apply_pilot_cap(
        full_pairs[full_pairs.endpoint.eq(primary_endpoint)].copy(), config
    )
    external_pairs = full_pairs[full_pairs.endpoint.eq(external_endpoint)].sort_values("sample_id").reset_index(drop=True)
    write_table(measurements, root / "data/processed/measurements.parquet")
    # The primary modeling table is exact IC50 only. Ki remains available as a
    # separate endpoint table and is never mixed into the IC50 benchmark.
    write_table(primary_pairs, root / "data/processed/pairs.parquet")
    if len(external_pairs):
        write_table(external_pairs, root / "data/processed/pairs_Ki.parquet")
    write_table(primary_pairs.head(100), root / "outputs/audit/traceable_pairs_100.parquet")
    inventory = primary_pairs.groupby(["target_chembl_id", "endpoint", "assay_A", "assay_B"], sort=True).agg(
        shared_molecules=("chemical_identity", "nunique"), pair_rows=("sample_id", "nunique"),
        positive_rate=("raw_label_ge_1", "mean"), mean_abs_delta=("abs_delta", "mean")).reset_index()
    write_table(inventory, root / "outputs/audit/pair_inventory.parquet")
    dependence = primary_pairs[["chemical_identity", "target_chembl_id", "endpoint",
                             "pair_rows_per_identity", "pair_row_weight"]].drop_duplicates()
    write_table(dependence, root / "outputs/audit/pair_dependence.parquet")
    if len(external_pairs):
        external_dependence = external_pairs[["chemical_identity", "target_chembl_id", "endpoint",
                                              "pair_rows_per_identity", "pair_row_weight"]].drop_duplicates()
        write_table(external_dependence, root / "outputs/audit/pair_dependence_Ki.parquet")
    target_inventory = measurements.groupby(["target_chembl_id", "standard_type"], sort=True).agg(
        measurements=("measurement_id", "nunique"), identities=("chemical_identity", "nunique"),
        assays=("assay_chembl_id", "nunique"), documents=("assay_doc_id", "nunique")).reset_index()
    write_table(target_inventory, root / "outputs/audit/target_inventory.csv")
    extraction, curation = {}, {}
    import json
    for name, path in [("extraction", root / "outputs/audit/extraction_counts.json"),
                       ("curation", root / "outputs/audit/curation_counts.json")]:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                (extraction if name == "extraction" else curation).update(json.load(handle))
    audit = audit_pairs(primary_pairs, measurements, extraction, curation)
    audit["primary_endpoint"] = primary_endpoint
    audit["external_endpoint"] = external_endpoint
    audit["primary_pair_rows"] = int(len(primary_pairs))
    audit["external_pair_rows"] = int(len(external_pairs))
    audit["all_endpoint_pair_rows"] = int(len(full_pairs))
    audit["pilot_selected_pair_rows"] = int(len(primary_pairs))
    audit["pilot_cap"] = cap_info
    dump_json(audit, root / "outputs/audit/pair_counts.json")
    manifest_files = [root / "data/processed/measurements.parquet", root / "data/processed/pairs.parquet",
                      root / "outputs/audit/traceable_pairs_100.parquet", root / "outputs/audit/pair_inventory.parquet",
                      root / "outputs/audit/pair_dependence.parquet",
                      root / "outputs/audit/target_inventory.csv"]
    if len(external_pairs):
        manifest_files += [root / "data/processed/pairs_Ki.parquet", root / "outputs/audit/pair_dependence_Ki.parquet"]
    dump_json({"release": int(config["project"]["release"]), "plan_version": config["project"]["plan_version"],
               "source_mode": extraction.get("source_mode", "sqlite"),
               "raw_database": extraction.get("database_path", str(root / "data/raw/chembl_37.db")),
               "primary_endpoint": primary_endpoint, "external_endpoint": external_endpoint,
               "immutable_raw_policy": True, "files": {str(path.relative_to(root)): sha256(path) for path in manifest_files},
               "counts_file": "outputs/audit/pair_counts.json", "split_manifest": "outputs/manifests/split_manifest.json",
               "pilot_cap": cap_info,
               "pair_dependence_file": "outputs/audit/pair_dependence.parquet"}, root / "outputs/manifests/dataset_manifest.json")
    lines = ["# ChEMBL37 Assay Disagreement Data Audit", "",
             "This report is generated from local outputs; no planning values are inserted.", "", "## Counts"]
    excluded = {"abs_delta_summary", "abs_delta_threshold_fraction", "shared_identity_bins",
                "source_type_fraction", "pilot_cap", "excluded_by_reason", "endpoints"}
    for key, value in audit.items():
        if key not in excluded:
            lines.append(f"- {key}: {value}")
    for title, values in [("Excluded by reason", audit.get("excluded_by_reason", {})),
                          ("Endpoints", audit.get("endpoints", {})),
                          ("Shared identity bins", audit.get("shared_identity_bins", {})),
                          ("Absolute delta threshold fractions", audit.get("abs_delta_threshold_fraction", {})),
                          ("Source type fractions", audit.get("source_type_fraction", {})),
                          ("Absolute delta summary", audit.get("abs_delta_summary", {}))]:
        lines += ["", f"## {title}"] + [f"- {key}: {value}" for key, value in values.items()]
    lines += ["", "## Provenance", "- Every emitted pair stores both measurement IDs and primitive activity IDs.",
              "- Assay A/B are sorted by stable ChEMBL assay ID; the pair is unordered.",
              "- Pair rows are not independent observations; inverse pair-row weights are emitted and applied in model fitting, with dependence summarized separately."]
    (root / "outputs/audit/audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "outputs/audit/audit_report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(config)
    ensure_dirs(root)
    run(config, root)


if __name__ == "__main__":
    main()
