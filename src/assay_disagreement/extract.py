"""Extract a provenance-preserving, broad candidate table from ChEMBL37."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from .io import db_path, dump_json, ensure_dirs, load_config, project_root, sha256
from .schema import require_schema


SQL = """
SELECT
 act.activity_id, act.molregno, mol.chembl_id AS molecule_chembl_id,
 cs.canonical_smiles, act.assay_id, ass.chembl_id AS assay_chembl_id,
 ass.tid, tgt.chembl_id AS target_chembl_id, tgt.pref_name AS target_name,
 tgt.target_type, tgt.tax_id AS target_tax_id, ass.description AS assay_description,
 ass.assay_type, ass.bao_format, ass.confidence_score, act.standard_type,
 act.standard_relation, act.standard_value, act.standard_units, act.pchembl_value,
 act.potential_duplicate, act.data_validity_comment, ass.doc_id AS assay_doc_id,
 doc.year AS document_year, doc.doi AS document_doi
FROM activities AS act
JOIN assays AS ass ON act.assay_id = ass.assay_id
JOIN target_dictionary AS tgt ON ass.tid = tgt.tid
JOIN molecule_dictionary AS mol ON act.molregno = mol.molregno
JOIN compound_structures AS cs ON act.molregno = cs.molregno
LEFT JOIN docs AS doc ON ass.doc_id = doc.doc_id
WHERE tgt.target_type = ? AND tgt.tax_id = ? AND ass.confidence_score = ?
  AND act.standard_type IN (?, ?)
"""


def counts(conn: sqlite3.Connection, config: dict) -> dict[str, int]:
    data = config["data"]
    total = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    target = conn.execute("""SELECT COUNT(*) FROM activities act
      JOIN assays ass ON act.assay_id=ass.assay_id
      JOIN target_dictionary tgt ON ass.tid=tgt.tid
      WHERE tgt.target_type=? AND tgt.tax_id=? AND ass.confidence_score=?
        AND act.standard_type IN (?, ?)""", (data["target_type"], data["target_tax_id"],
        data["confidence_score"], data["endpoint_primary"], data["endpoint_external"])).fetchone()[0]
    return {"raw_activity_rows": total, "target_endpoint_candidate_rows": target}


def run(config: dict, root: Path) -> Path:
    database = db_path(root)
    if not database.exists():
        if config.get("pilot", {}).get("allow_api_source", False):
            from .extract_api import run as api_run
            return api_run(config, root)
        raise RuntimeError(f"ChEMBL37 SQLite database not found: {database}")
    require_schema(database)
    output = root / "data/interim/extracted_activities.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as conn, output.open("w", newline="", encoding="utf-8") as handle:
        cursor = conn.execute(SQL, (config["data"]["target_type"], config["data"]["target_tax_id"],
                                    config["data"]["confidence_score"], config["data"]["endpoint_primary"],
                                    config["data"]["endpoint_external"]))
        fields = [item[0] for item in cursor.description]
        writer = csv.writer(handle)
        writer.writerow(fields)
        extracted_rows = 0
        while rows := cursor.fetchmany(10000):
            writer.writerows(rows)
            extracted_rows += len(rows)
    with sqlite3.connect(database) as conn:
        audit = counts(conn, config)
    archive = root / "data/raw/chembl_37_sqlite.tar.gz"
    audit.update({
        "extracted_candidate_rows": extracted_rows,
        "sql_candidate_rows_without_structure_join": audit["target_endpoint_candidate_rows"],
        "release": int(config["project"]["release"]),
        "source_mode": "chembl37_sqlite_local",
        "database_path": str(database),
        "database_sha256": sha256(database),
        "archive_path": str(archive) if archive.exists() else "",
        "archive_sha256": sha256(archive) if archive.exists() else "",
        "extracted_csv": str(output),
    })
    dump_json(audit, root / "outputs/audit/extraction_counts.json")
    print(output)
    return output


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
