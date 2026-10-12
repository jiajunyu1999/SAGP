"""Apply predeclared measurement filters and construct identity tables."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dump_json, ensure_dirs, load_config, project_root, sha256, write_table
from .standardize import context_flags, identity_fields


UNIT_TO_NM = {"M": 1e9, "mM": 1e6, "uM": 1e3, "µM": 1e3, "nM": 1.0, "pM": 1e-3}


def to_nm(value, unit):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    factor = UNIT_TO_NM.get(str(unit))
    if factor is None or not math.isfinite(value) or value <= 0:
        return np.nan
    return value * factor


def _provenance_key(row: pd.Series) -> str:
    values = (row.get("assay_doc_id"), row.get("molecule_chembl_id"), row.get("standard_type"),
              row.get("standard_value_nm"), row.get("assay_description"))
    return "prov_" + hashlib.sha256(repr(values).encode("utf-8")).hexdigest()[:24]


def _provenance_keys(frame: pd.DataFrame) -> list[str]:
    """Create stable lineage keys without a row-wise pandas ``apply``."""
    columns = ["assay_doc_id", "molecule_chembl_id", "standard_type",
               "standard_value_nm", "assay_description"]
    values = zip(*(frame[column].tolist() for column in columns))
    return ["prov_" + hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:24]
            for value in values]


def curate_frame(raw: pd.DataFrame, config: dict,
                 identity_cache_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = config["data"]
    work = raw.copy()
    work["_input_row"] = np.arange(len(work), dtype=np.int64)
    work["standard_value_nm"] = [to_nm(v, u) for v, u in zip(work.standard_value, work.standard_units)]
    work["p_activity"] = 9.0 - np.log10(work["standard_value_nm"])
    descriptions = work["assay_description"].fillna("")
    context_cache = {value: context_flags(value) for value in pd.unique(descriptions)}
    context = pd.json_normalize(descriptions.map(context_cache))
    work = pd.concat([work.reset_index(drop=True), context], axis=1)
    relation_ok = work.standard_relation.eq(data["standard_relation"])
    endpoint_ok = work.standard_type.isin([data["endpoint_primary"], data["endpoint_external"]])
    assay_type_main = data.get("assay_type_main")
    assay_type_ok = (work.assay_type.eq(assay_type_main)
                     if assay_type_main else pd.Series(True, index=work.index))
    value_ok = work.standard_value_nm.notna()
    duplicate_ok = ~work["potential_duplicate"].fillna(0).astype(str).str.lower().isin({"1", "true", "yes"})
    context_ok = ~work["known_cell_or_organism_context"].fillna(False)
    validity_comment = work.get("data_validity_comment", pd.Series("", index=work.index))
    validity_comment = validity_comment.fillna("").astype(str).str.strip()
    allowed_validity_comments = {
        str(value).strip().lower()
        for value in data.get("allowed_data_validity_comments", [])
    }
    validity_comment_ok = (
        validity_comment.eq("")
        | validity_comment.str.lower().isin(allowed_validity_comments)
    )
    if not data.get("exclude_data_validity_comments", True):
        validity_comment_ok = pd.Series(True, index=work.index)
    confidence_ok = work["confidence_score"].eq(data["confidence_score"])
    # Standardizing every raw activity dominates runtime, although rows already
    # excluded by an endpoint, relation, duplicate, context, validity, or
    # confidence rule cannot enter the curated set. Restrict RDKit work to the
    # remaining candidates; earlier failures do not receive a structure-failure
    # diagnostic because their structure was intentionally not evaluated.
    pre_structure_ok = (endpoint_ok & assay_type_ok & relation_ok & value_ok
                        & duplicate_ok & validity_comment_ok & context_ok
                        & confidence_ok)
    identity_columns = ["canonical_isomeric_smiles", "chemical_identity",
                        "chemical_family", "scaffold", "standardization_status"]
    identities = pd.DataFrame(index=work.index, columns=identity_columns)
    candidate_smiles = work.loc[pre_structure_ok, "canonical_smiles"].fillna("")
    identity_cache: dict[str, dict] = {}
    if identity_cache_frame is not None and not identity_cache_frame.empty:
        required = ["canonical_smiles", *identity_columns]
        missing = set(required) - set(identity_cache_frame.columns)
        if missing:
            raise ValueError(f"identity cache misses columns: {sorted(missing)}")
        cached = identity_cache_frame.drop_duplicates("canonical_smiles").set_index("canonical_smiles")
        identity_cache.update({
            str(smiles): row.to_dict()
            for smiles, row in cached[identity_columns].iterrows()
        })
    for value in pd.unique(candidate_smiles):
        if value not in identity_cache:
            identity_cache[value] = identity_fields(value)
    identities.loc[pre_structure_ok, identity_columns] = pd.json_normalize(
        candidate_smiles.map(identity_cache)
    ).to_numpy()
    work = pd.concat([work.reset_index(drop=True), identities.reset_index(drop=True)], axis=1)
    structure_ok = (~pre_structure_ok) | work["chemical_identity"].notna()
    # A record can fail more than one rule.  The audit needs two separate
    # views: a mutually-exclusive primary reason for reconciliation, and a
    # non-exclusive diagnostic count showing every rule that was hit.
    checks = [
        ("endpoint", endpoint_ok),
        ("non_exact_relation", relation_ok),
        ("invalid_value_or_unit", value_ok),
        ("unparseable_structure", structure_ok),
        ("potential_duplicate", duplicate_ok),
        ("data_validity_comment", validity_comment_ok),
        ("known_cell_or_organism_context", context_ok),
        ("confidence_score", confidence_ok),
        ("assay_type", assay_type_ok),
    ]
    valid = pre_structure_ok & structure_ok
    curated = work.loc[valid].copy()
    curated.drop(columns=["_input_row"], inplace=True)
    curated["measurement_id"] = (
        "meas_" + curated["activity_id"].astype(str) + "_" + curated["assay_id"].astype(str)
    )
    curated["provenance_equivalence_group"] = _provenance_keys(curated)
    exclusions = []
    failed_rule_counts = {}
    for idx in work.index[~valid]:
        failed = [name for name, passed in checks if not bool(passed.loc[idx])]
        for name in failed:
            failed_rule_counts[name] = failed_rule_counts.get(name, 0) + 1
        exclusions.append({
            "input_row": int(work.at[idx, "_input_row"]),
            "activity_id": work.at[idx, "activity_id"],
            "reason": failed[0],
            "all_reasons": ";".join(failed),
        })
    exclusions_frame = pd.DataFrame(exclusions, columns=["input_row", "activity_id", "reason", "all_reasons"])
    mutually_exclusive_counts = exclusions_frame["reason"].value_counts().to_dict()
    excluded_rows = int(len(raw) - len(curated))
    if len(exclusions_frame) != excluded_rows:
        raise AssertionError("curation exclusion rows do not reconcile with candidate rows")
    audit = {
        "candidate_rows": int(len(raw)), "curated_rows": int(len(curated)),
        "excluded_rows": excluded_rows,
        "excluded_by_reason": mutually_exclusive_counts,
        "failed_rule_counts_nonexclusive": failed_rule_counts,
        "count_reconciliation": {
            "candidate_equals_curated_plus_excluded": int(len(raw)) == int(len(curated)) + excluded_rows,
            "candidate_rows": int(len(raw)),
            "curated_rows": int(len(curated)),
            "excluded_rows": excluded_rows,
        },
        "identity_count": int(curated.chemical_identity.nunique()),
        "family_count": int(curated.chemical_family.nunique()),
        "target_count": int(curated.target_chembl_id.nunique()),
        "assay_count": int(curated.assay_chembl_id.nunique()),
        "document_count": int(curated.assay_doc_id.nunique()),
        "endpoints": curated.standard_type.value_counts().to_dict(),
    }
    return curated, exclusions_frame, audit


def run(config: dict, root: Path) -> None:
    raw = pd.read_csv(root / "data/interim/extracted_activities.csv", low_memory=False)
    cache_path = root / "data/interim/identity_standardization_cache.parquet"
    identity_cache = pd.read_parquet(cache_path) if cache_path.exists() else None
    curated, exclusions, audit = curate_frame(raw, config, identity_cache)
    write_table(curated, root / "data/processed/curated_measurements.csv")
    write_table(exclusions, root / "data/processed/exclusions.csv")
    molecules = curated[["molecule_chembl_id", "canonical_isomeric_smiles", "chemical_identity",
                         "chemical_family", "scaffold", "standardization_status"]].drop_duplicates()
    assays = curated[["assay_chembl_id", "assay_id", "target_chembl_id", "target_name", "assay_description",
                      "assay_type", "bao_format", "standard_type", "assay_doc_id", "document_year",
                      "document_doi", "known_cell_or_organism_context", "variant_signal"]].drop_duplicates()
    write_table(molecules, root / "data/processed/molecules.csv")
    write_table(assays, root / "data/processed/assays.csv")
    audit["identity_standardization_cache"] = {
        "used": identity_cache is not None,
        "path": str(cache_path) if identity_cache is not None else "",
        "sha256": sha256(cache_path) if identity_cache is not None else "",
        "rows": int(len(identity_cache)) if identity_cache is not None else 0,
        "note": "Deterministic SMILES-to-identity cache; missing entries are standardized from raw SMILES.",
    }
    dump_json(audit, root / "outputs/audit/curation_counts.json")
    print(root / "outputs/audit/curation_counts.json")


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
