# Additional experiment summary

Fixed five original scaffold splits and model checkpoints. Five frozen-model diagnostics, one pair-stratified selective analysis, and two reference interventions. Full diagnostic comparison includes six models. No refitting, test-time calibration fitting, or model selection. All unfavorable results retained. These are exploratory analyses on the existing benchmark.

## Reference interventions

|Task|Reference setting|AP|Pair AUROC|Macro MAE|
|---|---|---|---|---|
|1|3|0.3712 ± 0.1301|0.7275 ± 0.0556|0.3726 ± 0.0185|
|1|10|0.4944 ± 0.1545|0.7295 ± 0.0569|0.3421 ± 0.0196|
|1|30|0.5360 ± 0.1477|0.7379 ± 0.0533|0.3265 ± 0.0224|
|1|100|0.5395 ± 0.1481|0.7459 ± 0.0554|0.3212 ± 0.0257|
|1|Shuffle100|0.3304 ± 0.0888|0.7086 ± 0.0602|0.4003 ± 0.0379|
|2|3|0.4442 ± 0.0550|0.6240 ± 0.0614|0.2167 ± 0.0306|
|2|10|0.5540 ± 0.0493|0.6429 ± 0.0523|0.1886 ± 0.0216|
|2|30|0.6253 ± 0.0384|0.6760 ± 0.0335|0.1664 ± 0.0154|
|2|100|0.6439 ± 0.0422|0.6969 ± 0.0384|0.1602 ± 0.0114|
|2|Shuffle100|0.4548 ± 0.0445|0.5270 ± 0.0334|0.2112 ± 0.0315|

## Paired component intervals

|Task|Control|Metric|100-reference difference|95% CI|
|---|---|---|---|---|
|1|nested 3|AP|+0.1683|[+0.0801, +0.2664]|
|1|nested 3|pair_macro_AUROC|+0.0185|[+0.0024, +0.0338]|
|1|nested 3|macro_MAE|-0.0514|[-0.0779, -0.0341]|
|1|nested 30|AP|+0.0035|[-0.0025, +0.0131]|
|1|nested 30|pair_macro_AUROC|+0.0080|[-0.0003, +0.0167]|
|1|nested 30|macro_MAE|-0.0053|[-0.0100, -0.0026]|
|1|shuffle_shift 100|AP|+0.2091|[+0.0827, +0.2905]|
|1|shuffle_shift 100|pair_macro_AUROC|+0.0373|[+0.0126, +0.0596]|
|1|shuffle_shift 100|macro_MAE|-0.0791|[-0.1403, -0.0439]|
|2|nested 3|AP|+0.1997|[+0.1494, +0.2362]|
|2|nested 3|pair_macro_AUROC|+0.0729|[+0.0314, +0.1037]|
|2|nested 3|macro_MAE|-0.0566|[-0.0822, -0.0433]|
|2|nested 30|AP|+0.0186|[+0.0027, +0.0318]|
|2|nested 30|pair_macro_AUROC|+0.0209|[-0.0006, +0.0380]|
|2|nested 30|macro_MAE|-0.0062|[-0.0102, -0.0037]|
|2|shuffle_shift 100|AP|+0.1891|[+0.1260, +0.2509]|
|2|shuffle_shift 100|pair_macro_AUROC|+0.1699|[+0.1133, +0.2193]|
|2|shuffle_shift 100|macro_MAE|-0.0511|[-0.0977, -0.0268]|

Intervals reuse the original2000 shared assay/document-component bootstrap draws with fixed five-seed weighting. They are pointwise fitted-model conditional and do not include support-order or reference-shuffle uncertainty. One realization per split is insufficient for a permutation p-value. Task2 labels and pair offset use the full historical training pool at every count, so this does not establish end-to-end three-measurement adaptation.

## What the evidence can address
Reference interventions test whether historical molecule-response associations matter, while retaining the same query cohort. Dose response tests whether additional references help the frozen model. Probability diagnostics test reliability, not just ranking. Target-frequency removal and similarity bins describe applicability, without establishing unseen-target generalization. Pair-stratified retention keeps all assay pairs represented and scores every comparator on the same queries.

## Probability reliability

                             Brier   ECE10
task model                                
1    context_neural         0.0998  0.0444
     detached_action        0.0989  0.0424
     fixed_tanimoto_gp      0.1081  0.0743
     joint_monotone         0.0994  0.0403
     local_knn              0.1176  0.0839
     random_forest          0.1039  0.0709
     uncalibrated_gaussian  0.0992  0.0399
2    context_neural         0.0757  0.0279
     detached_action        0.0832  0.0546
     fixed_tanimoto_gp      0.1072  0.1484
     joint_monotone         0.0763  0.0317
     local_knn              0.0825  0.0294
     random_forest          0.0831  0.0336
     uncalibrated_gaussian  0.0750  0.0327

## Equal-pair selective prediction
Posterior-variance retention; every assay pair remains represented. Query class prevalence can still change.

task  budget
1     0.2       0.3046
      0.4       0.3089
      0.6       0.3132
      0.8       0.3184
      1.0       0.3212
2     0.2       0.1526
      0.4       0.1528
      0.6       0.1556
      0.8       0.1569
      1.0       0.1602

## Remaining high-value experiments
Repeat reference shuffles/support selection across multiple random realizations; retrain with within-pair shuffled molecule labels; evaluate frozen preprocessing on an external/temporal/target-disjoint cohort or Ki. Those experiments have not been run.

Detailed reports: REPORT.md, reference_interventions/REPORT.md, within_pair_selective_REPORT.md. CSVs retain individual seeds, query support metadata and all matched pair errors.
