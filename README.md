# SAGP

Learning Molecular Response Landscapes Across Bioassays

This folder corresponds to the current manuscript: two tasks and six main figures. Public model name: SAGP. Internal module/checkpoint names are retained so that serialized models load correctly.

- `paper/`: current main manuscript, Supplementary Information, TeX, bibliography and original unchanged template.
- `analysis/`: training, evaluation, plotting and their local code dependencies.
- `src/assay_disagreement/`: extraction/curation dependencies.
- Original ChEMBL37 SQLite database is excluded. Training/evaluation use the included processed data. To rerun extraction or raw-database case verification, place the downloaded ChEMBL37 database at `data/raw/chembl_37/chembl_37_sqlite/chembl_37.db`.
- `data/interim/` and `data/processed/`: extraction and curated records, molecular structures, assay metadata and paired measurements.
- `outputs/datasets/sar_transfer_20261007/`: exact task cohorts, five scaffold splits, reference manifests, feature cache, model weights and per-query predictions.
- `outputs/figures/`: six publication figures and their numerical source tables. `figure_data_index.json` maps each figure to its plotting script and data.
- `model_index.csv`: checkpoint/preprocessor inventory, with SAGP selected and final weights for all five seeds.
- `verification/`: reference audit, recomputed metric comparisons, reloaded-checkpoint prediction checks and file hashes.
- `source_provenance.json`: original source of collected files. Large files are losslessly compressed under `data_archives/` in blocks no larger than 32 MiB; `python restore_data.py` restores their original paths and verifies SHA-256 checksums. `restored_file_manifest.csv` describes the original restored files; `file_manifest.csv` describes the published tree.

## Manuscript version and access

The manuscript-aligned release is tagged [`v1.0.1`](https://github.com/jiajunyu1999/SAGP/tree/v1.0.1). Original data: [ChEMBL 37, DOI 10.6019/CHEMBL.database.37](https://doi.org/10.6019/CHEMBL.database.37). ChEMBL-derived data retain [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Files are publicly downloadable without registration or approval; the original database is obtained separately from EMBL–EBI. This GitHub release has no separately assigned DOI.

## Key implementation settings

SAGP uses Python 3.11, PyTorch 2.4.0 and RDKit 2025.3.6. Full versions are in `requirements-lock.txt` and `environment_versions.json`.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| GP learning rate / weight decay | 0.0003 / 0.003 |
| Decision-adapter learning rate / weight decay | 0.003 / 0 |
| Batch size / clipping norm | 32 / 5 |
| Training updates | 1,000 |
| Structured mean | Eight knots spanning −1.75 to 1.75 |
| Training references per episode | Sampled from 1, 3, 5, 10, 20, 40 |
| Evaluation references | At most 100 per assay pair; fixed random state 17 |
| Scaffold-split seeds | 17, 23, 41, 71, 101 |
| Checkpoint candidates | Initialization and every 100 updates |
| Validation score | Mean of Task 1/2 AP and within-pair AUROC, minus 0.1 × the sum of Task 1/2 macro MAE |

## Reproduce existing results

Use Python 3.11 and the versions in `requirements-lock.txt`. Run commands from this folder; preserve the original numerical environment when reconstructing classical controls.

```bash
python restore_data.py
pip install -r requirements-lock.txt
PYTHONPATH=src python analysis/recompute_paper_metrics_20261012.py
python analysis/verify_sagp_release_predictions_20261012.py
```

The checkpoint check reloads each selected SAGP fit and compares predictions with its archived test predictions. The metric check evaluates only models used in the current manuscript or its response analysis, on identical task-specific query cohorts. Main-text model comparisons and Figure 5 response formulations remain distinct analyses.

## Rebuild figures

```bash
python analysis/redraw_figure1_two_tasks_20261012.py
python analysis/make_two_task_figure2_20261012.py
python analysis/make_unfiltered_main_figure3_20261012.py
python analysis/redraw_combined_casebook_20261012.py
python analysis/make_merged_reference_response_20261012.py
python analysis/make_reliability_figure6_20261012.py
```

Cases use archived selections and measured structures. Figure 5a/b violin densities use five actual scaffold evaluations; Figure 5c uses 2,000 existing paired bootstrap draws per task/metric, with original estimates and 95% intervals. Boxplots in Figure 5g-i use five actual scaffold evaluations per task. Retrospective cases are illustrations rather than independent validation.

## Training and manuscript compilation

SAGP initialization weights, the exact reference manifests and data are included. The original trainer supports:

```bash
python analysis/train_contextual_decision_gp_20261009.py --models context_neural --seeds 17 23 41 71 101 --steps 1000
```

Existing completion markers cause the original trainer to skip finished runs. Use a separate copy of the folder for new training and retain the archived results as the comparison reference. Constant/prior and local kNN controls use historical data rather than neural weights. Linear and forest estimators were reconstructed with the original settings and accepted only after matching original validation and test predictions.

Compile the main manuscript and Supplementary Information into `paper/build/`, run `analysis/audit_manuscript_references_20261012.py` to regenerate SI table numbers, then recompile the main manuscript. The audit expects AUX files in `paper/build/`. The two remaining supplementary tables are S1 (validation-selected training variants) and S2 (final-step robustness). Duplicate main-text baseline tables, Figure 3 and core implementation equations were removed from SI.
