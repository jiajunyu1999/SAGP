# Controlled Joint GP ablation

All five original scaffold splits, identical query/support episode streams, identical test cohorts, root-supervision warm starts and preprocessing. Each variant completes 1,000 updates; checkpoint selection uses the same frozen validation rule. Final-step results test dependence on checkpoint selection.

A/B vs C tests auxiliary-supervision benefits, not equal-label information. The multihead control has the same GP/reference encoder and 58 additional parameters (3.9%); task outputs are independent learned moment heads. This is a parameter-sharing control, not a GP-free neural baseline. D removes the chemical kernel term; E permits signed response slopes with unchanged network size; F omits contrast covariance in both training and inference.

## Selected-checkpoint results

|Task|Model|AP|Within-pair AUROC|Macro MAE|
|---|---|---|---|---|
|1|task1_only|0.5382 ± 0.1511|0.7425 ± 0.0548|0.3218 ± 0.0263|
|1|task2_only|0.5398 ± 0.1523|0.7439 ± 0.0524|0.3216 ± 0.0269|
|1|joint|0.5404 ± 0.1493|0.7432 ± 0.0555|0.3216 ± 0.0263|
|1|no_molecular_residual|0.5309 ± 0.1486|0.7267 ± 0.0512|0.3292 ± 0.0218|
|1|no_monotone_mean|0.5411 ± 0.1492|0.7439 ± 0.0552|0.3221 ± 0.0270|
|1|no_contrast_covariance|0.5392 ± 0.1485|0.7437 ± 0.0557|0.3224 ± 0.0266|
|1|shared_multihead|0.5347 ± 0.1517|0.7406 ± 0.0540|0.3242 ± 0.0268|
|2|task1_only|0.6459 ± 0.0459|0.6980 ± 0.0357|0.1773 ± 0.0198|
|2|task2_only|0.6408 ± 0.0399|0.6947 ± 0.0335|0.1717 ± 0.0200|
|2|joint|0.6389 ± 0.0337|0.6978 ± 0.0331|0.1753 ± 0.0247|
|2|no_molecular_residual|0.6018 ± 0.0576|0.6319 ± 0.0314|0.1750 ± 0.0177|
|2|no_monotone_mean|0.6401 ± 0.0346|0.6981 ± 0.0341|0.1750 ± 0.0249|
|2|no_contrast_covariance|0.6471 ± 0.0469|0.6947 ± 0.0339|0.1681 ± 0.0158|
|2|shared_multihead|0.6466 ± 0.0491|0.6946 ± 0.0391|0.1643 ± 0.0159|

## Paired differences: Joint minus control

|Task|Control|Metric|Difference|95% CI|Seeds better|
|---|---|---|---|---|---|
|1|task1_only|AP|+0.0022|[-0.0014, +0.0063]|3/5|
|1|task1_only|pair_macro_AUROC|+0.0008|[-0.0014, +0.0037]|4/5|
|1|task1_only|macro_MAE|-0.0001|[-0.0013, +0.0013]|2/5|
|1|task2_only|AP|+0.0006|[-0.0031, +0.0055]|2/5|
|1|task2_only|pair_macro_AUROC|-0.0007|[-0.0061, +0.0043]|2/5|
|1|task2_only|macro_MAE|+0.0001|[-0.0017, +0.0018]|3/5|
|1|no_molecular_residual|AP|+0.0094|[+0.0019, +0.0287]|5/5|
|1|no_molecular_residual|pair_macro_AUROC|+0.0165|[+0.0029, +0.0342]|5/5|
|1|no_molecular_residual|macro_MAE|-0.0076|[-0.0191, +0.0008]|5/5|
|1|no_monotone_mean|AP|-0.0007|[-0.0027, +0.0035]|2/5|
|1|no_monotone_mean|pair_macro_AUROC|-0.0007|[-0.0028, +0.0025]|2/5|
|1|no_monotone_mean|macro_MAE|-0.0005|[-0.0014, +0.0002]|3/5|
|1|no_contrast_covariance|AP|+0.0012|[-0.0031, +0.0059]|3/5|
|1|no_contrast_covariance|pair_macro_AUROC|-0.0005|[-0.0044, +0.0037]|1/5|
|1|no_contrast_covariance|macro_MAE|-0.0007|[-0.0021, +0.0008]|2/5|
|1|shared_multihead|AP|+0.0057|[+0.0004, +0.0104]|4/5|
|1|shared_multihead|pair_macro_AUROC|+0.0026|[-0.0006, +0.0064]|4/5|
|1|shared_multihead|macro_MAE|-0.0025|[-0.0043, -0.0001]|3/5|
|2|task1_only|AP|-0.0070|[-0.0135, -0.0009]|2/5|
|2|task1_only|pair_macro_AUROC|-0.0002|[-0.0039, +0.0038]|2/5|
|2|task1_only|macro_MAE|-0.0021|[-0.0040, +0.0004]|4/5|
|2|task2_only|AP|-0.0019|[-0.0082, +0.0048]|2/5|
|2|task2_only|pair_macro_AUROC|+0.0031|[-0.0011, +0.0066]|4/5|
|2|task2_only|macro_MAE|+0.0036|[+0.0007, +0.0059]|1/5|
|2|no_molecular_residual|AP|+0.0371|[+0.0120, +0.0670]|4/5|
|2|no_molecular_residual|pair_macro_AUROC|+0.0660|[+0.0261, +0.0926]|5/5|
|2|no_molecular_residual|macro_MAE|+0.0002|[-0.0161, +0.0100]|4/5|
|2|no_monotone_mean|AP|-0.0012|[-0.0029, +0.0006]|4/5|
|2|no_monotone_mean|pair_macro_AUROC|-0.0003|[-0.0022, +0.0010]|4/5|
|2|no_monotone_mean|macro_MAE|+0.0003|[-0.0001, +0.0007]|1/5|
|2|no_contrast_covariance|AP|-0.0082|[-0.0172, -0.0013]|1/5|
|2|no_contrast_covariance|pair_macro_AUROC|+0.0031|[-0.0023, +0.0082]|4/5|
|2|no_contrast_covariance|macro_MAE|+0.0071|[+0.0021, +0.0117]|0/5|
|2|shared_multihead|AP|-0.0077|[-0.0170, +0.0002]|3/5|
|2|shared_multihead|pair_macro_AUROC|+0.0032|[-0.0051, +0.0148]|3/5|
|2|shared_multihead|macro_MAE|+0.0110|[+0.0019, +0.0191]|0/5|

Intervals resample connected assay/document dependency components with one shared weight vector across all five seeds, both tasks, all models and checkpoints. Seeds are dependent repeats, not independent observations. AP is pooled within each seed then averaged across seeds; within-pair AUROC requires ≥10 queries and ≥3 per class; MAE is pair-macro then seed-macro. An unevaluable seed invalidates that metric draw instead of changing seed weighting. Intervals are conditional on the fitted models; no multiple-comparison correction is applied.

All seed and pair outcomes, including negative differences, are retained. The matched-label comparison and covariance ablation determine whether a shared posterior is helpful; this experiment cannot establish that it is necessary.

## What the controlled experiment supports

Task 1, Joint minus Task1-only: AP +0.0022 (95% CI -0.0014 to +0.0063); within-pair AUROC +0.0008 (95% CI -0.0014 to +0.0037). Both intervals cross zero, and the final-checkpoint intervals also cross zero.
Task 2, Joint minus Task2-only: AP -0.0019 (95% CI -0.0082 to +0.0048); within-pair AUROC +0.0031 (95% CI -0.0011 to +0.0066). Neither gives a confirmed ranking gain. Macro MAE is worse by +0.0036 (95% CI +0.0007 to +0.0059); this disadvantage persists at the final checkpoint.
The shared-head control gives a small selected-checkpoint Task 1 AP difference +0.0057 (95% CI +0.0004 to +0.0104), but its final-checkpoint AP interval crosses zero. It also gives better Task 2 magnitude prediction: Joint minus shared-head MAE +0.0110 (95% CI +0.0019 to +0.0191). This is not stable evidence that posterior sharing is necessary.
The strongest supported component is the molecular Tanimoto residual. Removing it reduces Task 1 within-pair AUROC by +0.0165 (95% CI +0.0029 to +0.0342) and Task 2 within-pair AUROC by +0.0660 (95% CI +0.0261 to +0.0926). AP and within-pair AUROC advantages over this ablation remain positive under both checkpoint rules for both tasks.

Recommended claim: the reference-conditioned chemical residual supports cross-assay molecular ranking. A common posterior provides coherent contrast and point readouts, but these results do not demonstrate a required multi-task learning benefit. Do not attribute the earlier Joint-versus-Conditional difference to joint training alone.

Root/split/source controls and numerical reproduction: all35 fits used1,000 updates and exact matched episode hashes within seed. Relative to the archived full Joint run, maximum absolute AP deviation is about0.000012 and within-pair AUROC deviation about0.000010; see joint_reproduction_difference_from_original.csv.

## Frozen covariance diagnostic

The exact same trained Joint checkpoints were evaluated after dropping only the contrast covariance term. Signed means and gap MAE remain identical. At selected checkpoints, Joint-minus-independent within-pair AUROC is−0.0032 (95% CI−0.0060 to−0.0005); the final-step interval also favors independent variance. Global AP intervals cross zero. Thus covariance is not supported as a classification-ranking gain.

The covariance readout gives nominal90% intervals with observed pair/seed-macro coverage89.92%, versus92.23% for independent variance, and mean halfwidth0.6816 versus0.7689 pIC50. At final checkpoints coverage is91.16% versus93.22%, and halfwidth0.7013 versus0.7817. This is a descriptive calibration/precision benefit; Brier and gap-NLL difference intervals cross zero. All calibration and ranking results are retained.
