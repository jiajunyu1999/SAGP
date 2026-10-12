Controlled Joint-versus-Single-task GP Ablation

Train (35 fits, five original scaffold seeds):
PYTHONPATH=src python analysis/controlled_joint_ablation_20261009.py
Aggregate all seeds with 2000 shared dependency-component draws:
PYTHONPATH=src python analysis/summarize_controlled_joint_ablation_20261009.py
Plot:
PYTHONPATH=src python analysis/plot_controlled_joint_ablation_20261009.py
Frozen inference-only covariance diagnostic:
PYTHONPATH=src python analysis/diagnose_frozen_contrast_covariance_20261009.py
Standalone appendix:
PYTHONPATH=src python analysis/make_controlled_ablation_appendix_20261009.py

Run from the assay-disagreement-plan project root. Completed fits are resumable; --overwrite deliberately reruns requested fits. Keep the archived input and checkpoint hashes. The five seeds can run on separate GPUs using CUDA_VISIBLE_DEVICES.

REPORT.md gives all selected results and paired CIs; test_summary.csv includes selected and final-budget checkpoints; paired_component_bootstrap.csv includes all72 differences. metrics_by_seed.csv and metrics_by_seed_and_pair.csv retain all signs and outcomes. reference_manifest_seed_*.csv fixes deployment references, root_training_membership_seed_*.csv records the complete training pool, compute_and_control_audit.csv verifies identical episode hashes and budgets. Frozen inference-only covariance results are separate from retrained ablationF.

A/B versusC compares auxiliary losses, not identical task-supervision types. G matches the task/root supervision, uses the same GP encoder with independent learned heads, and has58 additional parameters; it is not a GP-free neural baseline. D/E/F are structural ablations with the same nominal GP capacity except intentionally removed terms. Figures1-4 and the original main manuscript are preserved; the new appendix is separate.
