"""Schema probing and strict ChEMBL37 SQLite adapter."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .io import db_path, dump_json, ensure_dirs, load_config, project_root


REQUIRED = {
    "activities": {"activity_id", "molregno", "assay_id", "standard_type", "standard_relation",
                    "standard_value", "standard_units"},
    "assays": {"assay_id", "chembl_id", "tid", "description", "assay_type", "doc_id"},
    "target_dictionary": {"tid", "chembl_id", "pref_name", "target_type", "tax_id"},
    "molecule_dictionary": {"molregno", "chembl_id"},
    "compound_structures": {"molregno", "canonical_smiles"},
    "docs": {"doc_id", "year", "doi"},
}
API_ACTIVITY_FIELDS = ["activity_id", "molecule_chembl_id", "canonical_smiles", "assay_chembl_id",
                       "target_chembl_id", "standard_type", "standard_relation", "standard_value",
                       "standard_units"]


def inspect_schema(path: Path) -> dict:
    if not path.exists():
        return {"status": "unavailable", "path": str(path), "reason": "database_not_found"}
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {}
        for table in sorted(tables):
            columns[table] = [row[1] for row in conn.execute(f"PRAGMA table_info([{table}])")]
    missing = {table: sorted(REQUIRED[table] - set(columns.get(table, [])))
               for table in REQUIRED if table not in tables or REQUIRED[table] - set(columns.get(table, []))}
    return {"status": "ok" if not missing else "schema_incompatible", "path": str(path),
            "tables": sorted(tables), "columns": columns, "missing_required": missing}


def require_schema(path: Path) -> dict:
    report = inspect_schema(path)
    if report["status"] != "ok":
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root(config)
    ensure_dirs(root)
    report = inspect_schema(db_path(root))
    extraction_manifest = root / "outputs/audit/extraction_counts.json"
    if report["status"] == "unavailable" and extraction_manifest.exists():
        with extraction_manifest.open(encoding="utf-8") as handle:
            source = json.load(handle)
        if source.get("source_mode") == "chembl37_api_bounded_pilot":
            report = {"status": "api_adapter_verified", "release": 37,
                      "database": str(db_path(root)), "sqlite_status": "unavailable",
                      "api_status": source.get("api_status"),
                      "adapter_fields": API_ACTIVITY_FIELDS}
    dump_json(report, root / "outputs/audit/schema_report.json")
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
