# Frozen reference interventions

Fixed five splits, selected model weights, test queries and labels. No retraining or selection on these results. References use a deterministic shuffled ordering of the original capped training-only support, with nested sets3/10/30/100. At100 the original matrix ordering is restored to reproduce the exact original numerical computation; all smaller supports are subsets of the100pool. The100reference predictions reproduce original predictions to tolerance2e-4. Groups with fewer observations retain all available references. Shuffle control permutes signed responses among historical compounds within each assay pair at the same100cap, retaining the marginal shift distribution and source potencies. This is a fitted-model information intervention, not a training permutation null or a causal biological experiment.

Task2 training-only pair median and labels stay fixed across reference counts; they still use the original full training pool. Therefore this probes posterior/adaptor reference dependence, not end-to-end learning from onlykmeasurements. Task1 support-derived replicate width changes with the selected references. SD summarizes dependent splits, not confidence intervals. Only one deterministic support/shuffle realization per split was run; reference-selection stability is unmeasured.

                               AP         pair_macro_AUROC         macro_MAE        
                             mean     std             mean     std      mean     std
task condition     budget                                                           
1    nested        3       0.3712  0.1301           0.7275  0.0556    0.3726  0.0185
                   10      0.4944  0.1545           0.7295  0.0569    0.3421  0.0196
                   30      0.5360  0.1477           0.7379  0.0533    0.3265  0.0224
                   100     0.5395  0.1481           0.7459  0.0554    0.3212  0.0257
     shuffle_shift 100     0.3304  0.0888           0.7086  0.0602    0.4003  0.0379
2    nested        3       0.4442  0.0550           0.6240  0.0614    0.2167  0.0306
                   10      0.5540  0.0493           0.6429  0.0523    0.1886  0.0216
                   30      0.6253  0.0384           0.6760  0.0335    0.1664  0.0154
                   100     0.6439  0.0422           0.6969  0.0384    0.1602  0.0114
     shuffle_shift 100     0.4548  0.0445           0.5270  0.0334    0.2112  0.0315
