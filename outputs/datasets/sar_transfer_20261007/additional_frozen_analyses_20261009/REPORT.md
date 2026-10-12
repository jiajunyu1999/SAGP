# Frozen-model additional analyses

Six models, all five original scaffold splits. Models and query outcomes are unchanged. Strong controls are included regardless of their mean ranking. These are post-hoc descriptive diagnostics, not independent validation or causal experiments. SD reflects dependent scaffold evaluations, not confidence intervals.

## Probability calibration
ECE uses ten equal-width bins; Brier and log loss are evaluated without recalibration on test data.

                             Brier  log_loss   ECE10
task model                                          
1    context_neural         0.0998    0.3232  0.0444
     detached_action        0.0989    0.3190  0.0424
     fixed_tanimoto_gp      0.1081    0.3646  0.0743
     joint_monotone         0.0994    0.3254  0.0403
     local_knn              0.1176    0.7045  0.0839
     random_forest          0.1039    0.3489  0.0709
     uncalibrated_gaussian  0.0992    0.3236  0.0399
2    context_neural         0.0757    0.2478  0.0279
     detached_action        0.0832    0.2766  0.0546
     fixed_tanimoto_gp      0.1072    0.3787  0.1484
     joint_monotone         0.0763    0.2544  0.0317
     local_knn              0.0825    0.3751  0.0294
     random_forest          0.0831    0.2687  0.0336
     uncalibrated_gaussian  0.0750    0.2506  0.0327

## Applicability domain
Nearest-reference Tanimoto uses the exact training-only reference manifest capped at100. Task1 uses the smaller of the two query molecules’ nearest similarities. Bin membership is fixed without outcome access. AUROC requires both classes and pair AUROC retains the original >=10 queries / >=3 per class rule. Empty/one-class bins remain missing.

                   n      AP   AUROC  macro_MAE
task sim_bin                                   
1    0.3–0.5    19.6  0.6395  0.7255     0.4431
     0.5–0.7  1325.4  0.4888  0.7874     0.3042
     0.7–0.8  4181.0  0.5314  0.8391     0.3152
     0.8–1.0  1438.6  0.6362  0.8688     0.3278
     0–0.3       0.4     NaN     NaN     0.5001
2    0.3–0.5   230.0  0.7321  0.8908     0.0983
     0.5–0.7   729.0  0.6817  0.9032     0.1621
     0.7–0.8  1385.8  0.6535  0.8863     0.1437
     0.8–1.0   756.8  0.5659  0.8608     0.1515
     0–0.3      41.0  0.8874  0.8529     0.3657

## Target concentration
The top10 targets are defined per split from the capped training-reference pool. Removing their test queries is a robustness diagnostic, not target-disjoint training.

                         n  targets      AP  pair_macro_AUROC  macro_MAE
task subset                                                             
1    all            6965.0     92.4  0.5395            0.7459     0.3212
     exclude_top10  2076.4     84.8  0.6897            0.7160     0.4078
     top10          4888.6      7.6  0.3481            0.7722     0.2387
2    all            3142.6     78.4  0.6439            0.6969     0.1602
     exclude_top10  1164.0     71.0  0.7080            0.7046     0.2241
     top10          1978.6      7.4  0.4236            0.6585     0.0993

## Selective prediction
Retention orders are fixed decision entropy, posterior signed-shift/contrast variance, and one deterministic random ordering per seed. MAE is micro to keep within-retained-sample interpretation. For Task2 the variance describes signed shift, not calibrated absolute sensitivity; low variance need not imply low magnitude error. Filtering changes pair composition and class prevalence and is not a matched comparison to full coverage. Low classification error may partly reflect enrichment of negative/easy queries, so it does not by itself establish calibration or within-pair utility.

                                  positive_rate  classification_error  micro_MAE
task ordering           coverage                                                
1    decision_entropy   0.2              0.1069                0.0100     0.2875
                        0.4              0.0825                0.0225     0.2939
                        0.6              0.0993                0.0478     0.3084
                        0.8              0.1258                0.0840     0.3265
                        1.0              0.1689                0.1377     0.3400
     posterior_variance 0.2              0.0688                0.0680     0.1837
                        0.4              0.0938                0.0898     0.2258
                        0.6              0.1049                0.1009     0.2494
                        0.8              0.1385                0.1218     0.2941
                        1.0              0.1689                0.1377     0.3400
     random             0.2              0.1689                0.1381     0.3384
                        0.4              0.1676                0.1338     0.3362
                        0.6              0.1677                0.1359     0.3370
                        0.8              0.1683                0.1378     0.3394
                        1.0              0.1689                0.1377     0.3400
2    decision_entropy   0.2              0.0411                0.0024     0.0693
                        0.4              0.0301                0.0051     0.0729
                        0.6              0.0388                0.0154     0.0849
                        0.8              0.0722                0.0476     0.1097
                        1.0              0.1363                0.1034     0.1464
     posterior_variance 0.2              0.0118                0.0120     0.0760
                        0.4              0.0209                0.0202     0.0785
                        0.6              0.0313                0.0297     0.0880
                        0.8              0.0693                0.0587     0.1085
                        1.0              0.1363                0.1034     0.1464
     random             0.2              0.1408                0.1065     0.1499
                        0.4              0.1353                0.1049     0.1489
                        0.6              0.1349                0.1027     0.1495
                        0.8              0.1375                0.1046     0.1486
                        1.0              0.1363                0.1034     0.1464

## Sensitivity thresholds
Task2 continuous magnitude output is ranked at predeclared0.3,0.5,1.0,1.5 pIC50 thresholds. This tests regression ranking across severities, not a classifier trained/calibrated for each threshold.

           positive      AP   AUROC
threshold                          
0.3           783.4  0.6836  0.8503
0.5           394.2  0.6421  0.8998
1.0           139.6  0.6631  0.9501
1.5            60.8  0.7167  0.9659

## Next controlled interventions
Reference-count dose response with frozen weights, matched random seeds and nested supports; matched reference-quality interventions at constant count; train-only permutation null with repeated refits; genuinely held-out target/source or Ki replication. Separate nested-support and reference-response permutation intervention results are recorded under reference_interventions/; refitting permutation nulls and external replication remain outstanding.
