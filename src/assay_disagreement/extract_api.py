"""Small, auditable ChEMBL37 API adapter used when the SQLite archive is absent.

The API is accepted only when its status endpoint reports ChEMBL_37. The pilot
is bounded to a predeclared target candidate list and records this limitation.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

from .io import dump_json


BASE = "https://www.ebi.ac.uk/chembl/api/data"


def get_json(path: str, params: dict) -> dict:
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    last_error = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "assay-disagreement/0.1"})
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ChEMBL API request failed after retries: {url}: {last_error}")


def fetch_activities(targets: list[str], config: dict) -> pd.DataFrame:
    data = config["data"]
    rows = []
    for target in targets:
        offset = 0
        print(f"fetching target={target}", flush=True)
        while True:
            payload = get_json("activity.json", {
                "limit": 1000, "offset": offset, "target_chembl_id": target,
                "standard_type": data["endpoint_primary"],
                "standard_relation": data["standard_relation"],
                "standard_units": data["standard_units"], "assay_type": data["assay_type_main"],
            })
            activities = payload.get("activities", [])
            rows.extend(activities)
            print(f"  offset={offset} rows={len(activities)}", flush=True)
            if not payload.get("page_meta", {}).get("next") or not activities:
                break
            offset += len(activities)
            time.sleep(0.05)
    return pd.DataFrame(rows)


def enrich_assays(targets: list[str], config: dict) -> dict[str, dict]:
    """Fetch the bounded confidence-filtered assay universe, not one request per ID."""
    data = config["data"]
    result = {}
    for target in targets:
        offset = 0
        while True:
            payload = get_json("assay.json", {"limit": 1000, "offset": offset,
                                                "target_chembl_id": target,
                                                "assay_type": data["assay_type_main"],
                                                "confidence_score": data["confidence_score"]})
            records = payload.get("assays", [])
            for assay in records:
                result[assay["assay_chembl_id"]] = assay
            print(f"assay metadata target={target} offset={offset} rows={len(records)}", flush=True)
            if not payload.get("page_meta", {}).get("next") or not records:
                break
            offset += len(records)
            time.sleep(0.05)
    return result


def run(config: dict, root: Path) -> Path:
    candidates = config.get("pilot", {}).get("target_candidates", [])
    if not candidates:
        raise RuntimeError("pilot.target_candidates is required for bounded API extraction")
    status = get_json("status.json", {})
    if status.get("chembl_db_version") != "ChEMBL_37":
        raise RuntimeError(f"API is not fixed ChEMBL37: {status}")
    cap = int(config.get("pilot", {}).get("initial_target_cap", len(candidates)))
    data = config["data"]
    target_counts = {}
    for target in candidates:
        payload = get_json("activity.json", {
            "limit": 1, "target_chembl_id": target,
            "standard_type": data["endpoint_primary"],
            "standard_relation": data["standard_relation"],
            "standard_units": data["standard_units"], "assay_type": data["assay_type_main"],
        })
        target_counts[target] = int(payload.get("page_meta", {}).get("total_count", 0))
    selected = [target for target, _ in sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))[:cap]]
    activities = fetch_activities(selected, config)
    assays = enrich_assays(selected, config) if len(activities) else {}
    output_rows = []
    for _, row in activities.iterrows():
        assay = assays.get(row.assay_chembl_id, {})
        output_rows.append({
            "activity_id": row.activity_id, "molregno": row.molecule_chembl_id,
            "molecule_chembl_id": row.molecule_chembl_id, "canonical_smiles": row.canonical_smiles,
            "assay_id": row.assay_chembl_id, "assay_chembl_id": row.assay_chembl_id,
            "tid": row.target_chembl_id, "target_chembl_id": row.target_chembl_id,
            "target_name": row.target_pref_name, "target_type": "SINGLE PROTEIN",
            "target_tax_id": row.target_tax_id, "assay_description": row.assay_description,
            "assay_type": row.assay_type, "bao_format": row.bao_format,
            "confidence_score": assay.get("confidence_score"), "standard_type": row.standard_type,
            "standard_relation": row.standard_relation, "standard_value": row.standard_value,
            "standard_units": row.standard_units, "pchembl_value": row.pchembl_value,
            "potential_duplicate": row.potential_duplicate, "data_validity_comment": row.data_validity_comment,
            "assay_doc_id": row.document_chembl_id, "document_year": row.document_year,
            "document_doi": "", "api_assay_relationship_type": assay.get("relationship_type"),
        })
    frame = pd.DataFrame(output_rows)
    out = root / "data/interim/extracted_activities.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    dump_json({
        "source_mode": "chembl37_api_bounded_pilot", "api_status": status,
        "target_candidates": candidates, "selected_targets": selected,
        "eligible_candidate_counts": target_counts, "selection_rule": "highest API eligible row count, tie by target ID",
        "activity_rows_returned": int(len(frame)), "raw_activity_rows": int(len(frame)),
        "chembl37_global_activity_rows": int(status.get("activities", 0)),
        "assays_enriched": len(assays),
        "confidence_score_missing_assays": int(sum(a.get("confidence_score") is None for a in assays.values())),
        "limitation": "bounded target candidate extraction, not a complete SQLite snapshot",
        "output": str(out),
    }, root / "outputs/audit/extraction_counts.json")
    dump_json({
        "status": "api_adapter_verified", "release": 37,
        "api_status_endpoint": f"{BASE}/status.json",
        "required_activity_fields": sorted(frame.columns.tolist()),
        "required_assay_fields": ["assay_chembl_id", "confidence_score", "description",
                                   "assay_type", "bao_format", "document_chembl_id"],
        "sqlite_snapshot": "unavailable",
    }, root / "outputs/audit/schema_report.json")
    print(out)
    return out
