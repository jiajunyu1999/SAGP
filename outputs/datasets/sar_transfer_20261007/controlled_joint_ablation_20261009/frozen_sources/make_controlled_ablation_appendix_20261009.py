"""Standalone reviewable appendix; original manuscript figures stay untouched."""
from pathlib import Path
import pandas as pd
from controlled_joint_ablation_20261009 import RESULTS,MODELS
from plot_controlled_joint_ablation_20261009 import LABELS


def main():
    ms=pd.read_csv(RESULTS/'metrics_by_seed.csv').query('partition=="test"')
    ci=pd.read_csv(RESULTS/'paired_component_bootstrap.csv')
    paper=Path(__file__).resolve().parents[1]/'paper'
    text=[r'\documentclass[10pt]{article}',r'\usepackage[margin=0.65in]{geometry}',
          r'\usepackage[T1]{fontenc}',r'\usepackage{newtxtext,newtxmath,booktabs,graphicx,caption}',
          r'\usepackage[hidelinks]{hyperref}',r'\setlength{\emergencystretch}{1em}',
          r'\graphicspath{{../outputs/datasets/sar_transfer_20261007/controlled_joint_ablation_20261009/figures/}}',
          r'\title{Controlled Joint-versus-Single-task GP Ablation}',r'\author{}',r'\date{}',r'\begin{document}',r'\maketitle',
          r'\section*{Matched protocol}',
          'Seven variants reuse the five original scaffold splits (17, 23, 41, 71, 101), exactly the original validation/test query identifiers and labels, the same training-only preprocessing, and the same root-shift-pretrained conditional-monotone checkpoint within each seed. Each completes 1,000 AdamW updates (learning rate 0.0003; weight decay 0.003; gradient clipping 5), without early stopping. Training draws 32 queries per loss and reference budgets 1, 3, 5, 10, 20 or 40. Every variant draws all three streams even when a task loss is disabled; matching SHA256 hashes verify byte-identical episode tensors. Evaluation uses the original fixed, label-blind reference subset, capped at 100 per assay pair.',
          '',
          'A trains Task 1 plus root-shift losses; B trains Task 2 plus root-shift losses; C trains both tasks plus root-shift losses. D removes the Tanimoto residual kernel. E replaces positive response slopes with signed slopes, retaining the nine-slope network and smooth hinges. F omits the cross-molecule covariance when computing contrast variance during both training and inference. G uses the same GP/reference encoder with a small shared neural readout trunk and independent root/Task 1/Task 2 moment heads. G has 1,541 parameters versus 1,483 for A--F, so it is a slightly larger parameter-sharing control rather than a GP-free neural baseline. Structural ablations necessarily change their initial forward functions despite sharing checkpoint weights.',
          '',
          r'All task-loss coefficients remain fixed. A/B versus C tests additional task-loss supervision, not equality of supervision types. G versus C matches both task and root supervision. All variants use the same validation criterion: the mean of the two tasks\textquotesingle{} AP and within-pair AUROC, minus 0.1 times the sum of their macro MAEs. Thus both validation tasks select all variants, even A/B; final-budget checkpoints separately test dependence on this selection rule. Task 2 uses a folded-normal median for the continuous absolute-residual prediction.',
          '',r'\section*{Complete test results}',
          r'Values are mean $\pm$ split SD; SD is descriptive because the splits are dependent repeats. Pair AUROC requires at least ten queries and three per class. Macro MAE averages within assay pair, then over pairs.']
    names={m:LABELS[m].replace('  ',' ' ) for m in MODELS}
    for ck in ['selected','final']:
        text += [r'\begin{table}[ht]',r'\centering\small',r'\begin{tabular}{llccc}',r'\toprule',r'Task & Variant & AP & Pair AUROC & Macro MAE \\',r'\midrule']
        for task in [1,2]:
            for model in MODELS:
                f=ms.query('checkpoint==@ck and task==@task and model==@model')
                vals=[f'{f[k].mean():.4f} $\\pm$ {f[k].std():.4f}' for k in ['AP','pair_macro_AUROC','macro_MAE']]
                text.append(f'{task} & {names[model]} & '+' & '.join(vals)+r' \\')
            if task==1:text.append(r'\midrule')
        text += [r'\bottomrule\end{tabular}',r'\caption{'+('Unified validation-selected checkpoints.' if ck=='selected' else 'Final 1,000-update checkpoints; no checkpoint selection.')+'}',r'\end{table}']
    text += [r'\clearpage',r'\section*{Paired uncertainty and primary comparisons}',
             r'We resample the original connected assay/document dependency components 2,000 times. A single component-weight vector is shared across all five seeds, both tasks, all models and both checkpoint rules. The point estimate and bootstrap estimand retain equal seed weighting; an unevaluable seed invalidates that metric draw rather than changing the seed mixture. Intervals condition on fitted models and do not represent uncertainty from retraining. No multiple-comparison adjustment is applied; the full structural ablations are exploratory.',
             r'\begin{table}[ht]\centering\small',r'\begin{tabular}{llcrr}',r'\toprule',r'Task & Comparison & Metric & Joint minus control & 95\% interval \\',r'\midrule']
    for task,control in [(1,'task1_only'),(2,'task2_only'),(1,'shared_multihead'),(2,'shared_multihead')]:
        for key,label in [('AP','AP'),('pair_macro_AUROC','Pair AUROC'),('macro_MAE','Macro MAE')]:
            r=ci.query('checkpoint=="selected" and task==@task and control==@control and metric==@key').iloc[0]
            text.append(f'{task} & {names[control]} & {label} & {r.joint_minus_control:+.4f} & [{r.CI_low:+.4f}, {r.CI_high:+.4f}]'+r' \\')
    text += [r'\bottomrule\end{tabular}',r'\caption{Primary controlled comparisons. Positive ranking differences and negative error differences favor Joint GP.}',r'\end{table}',
             'Every seed and assay-pair outcome, including negative differences, is archived. The component-bootstrap intervals and final-checkpoint sensitivity analyses distinguish reproducible auxiliary-supervision gains from checkpoint-specific effects. A favorable A/B comparison alone cannot establish that a shared posterior is necessary; that interpretation additionally requires the matched-supervision head comparison and the covariance ablation.',
             r'\clearpage']
    primary=ci[((ci.task==1)&ci.control.eq('task1_only'))|((ci.task==2)&ci.control.eq('task2_only'))]
    ranking=primary[primary.metric.isin(['AP','pair_macro_AUROC'])]
    text += [r'\section*{Interpretation}']
    if ((ranking.CI_low<=0)&(ranking.CI_high>=0)).all():
        text += ['Neither primary single-task comparison establishes a ranking advantage for Joint GP: all AP and within-pair AUROC intervals cross zero under both checkpoint rules. These controlled results therefore do not demonstrate that adding the other task loss is necessary.']
    err=primary.query('checkpoint=="selected" and task==2 and metric=="macro_MAE"').iloc[0]
    text += [f'Task 2 macro MAE is higher for Joint GP than Task2-only by {err.joint_minus_control:.4f} pIC$_{{50}}$ (95\\% interval {err.CI_low:.4f}--{err.CI_high:.4f}); this error disadvantage also remains at the final checkpoint.']
    chemical=ci[ci.control.eq('no_molecular_residual')&ci.metric.isin(['AP','pair_macro_AUROC'])]
    if (chemical.CI_low>0).all():
        text += ['The most consistent supported component is the molecular Tanimoto residual: removing it worsens AP and within-pair AUROC for both tasks, with positive Joint-minus-ablation intervals under both checkpoint rules. The small selected-checkpoint Task 1 advantage over the shared independent-head control does not persist as a confirmed ranking advantage at the final checkpoint. Task 2 magnitude prediction favors the independent-head control. A defensible interpretation is therefore chemically local cross-assay response modeling with coherent posterior readouts, rather than a demonstrated requirement for multi-task optimization.']
    frozen=RESULTS/'frozen_covariance_paired_intervals.csv'
    if frozen.exists():
        text += [r'\section*{Frozen-posterior covariance diagnostic}',
                 'A separate inference-only diagnostic loads the exact same trained Joint GP checkpoints and removes only the contrast covariance term. Query means and continuous gap predictions are verified unchanged. Unlike F, this diagnostic does not retrain the model, so ranking changes isolate the variance readout rather than changes in learned parameters.',
                 r'\begin{table}[ht]\centering\small',r'\begin{tabular}{llrr}',r'\toprule',r'Checkpoint & Task 1 metric & Joint minus independent & 95\% interval \\',r'\midrule']
        for r in pd.read_csv(frozen).itertuples():
            key={'AP':'AP','pair_macro_AUROC':'Pair AUROC','macro_MAE':'Macro MAE'}[r.metric]
            text.append(f'{r.checkpoint} & {key} & {r.joint_minus_frozen_independent:+.4f} & [{r.CI_low:+.4f}, {r.CI_high:+.4f}]'+r' \\')
        text += [r'\bottomrule\end{tabular}',r'\caption{Variance-only readout diagnostic, using the same shared component-bootstrap draws.}',r'\end{table}',r'\clearpage']
        calibration=RESULTS/'frozen_covariance_calibration_by_seed.csv'
        if calibration.exists():
            f=pd.read_csv(calibration).groupby(['checkpoint','model']).mean(numeric_only=True)
            text += [r'\section*{Interval calibration in the frozen diagnostic}',
                r'\begin{table}[ht]\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',
                r'Checkpoint & Readout & Brier & Gap NLL & 90\% coverage & Halfwidth \\',r'\midrule']
            for ck in ['selected','final']:
                for model in ['joint','frozen_independent_variance']:
                    r=f.loc[ck,model]
                    name='Joint covariance' if model=='joint' else 'Independent variance'
                    text.append(f'{ck} & {name} & {r.pair_macro_Brier:.4f} & {r.pair_macro_gap_NLL:.4f} & {100*r.pair_macro_coverage_90:.2f}\\% & {r.pair_macro_halfwidth_90:.4f}'+r' \\')
            text += [r'\bottomrule\end{tabular}',r'\caption{Pair-macro, then seed-macro calibration statistics. Halfwidth is in pIC$_{50}$ units.}',r'\end{table}',
                'The covariance readout gives narrower intervals and empirical coverage closer to the nominal 90\% under both checkpoint rules. This descriptive interval-calibration benefit coexists with a small decrease in within-pair classification AUROC; the Brier and gap-NLL difference intervals include zero. It should not be presented as a confirmed classification-performance gain.',r'\clearpage']
    plots=[('controlled_ablation_seed_comparison','Seven variants with five seed dots, mean and split SD. Blue uses the common validation criterion; peach uses the final update.'),
           ('controlled_ablation_paired_intervals','Joint-minus-control differences with shared dependency-component bootstrap intervals for every ablation and both checkpoint rules. All differences are retained.'),
           ('controlled_ablation_unique_pair_changes','Each point is one unique assay pair, averaging its available selected-checkpoint seed differences. Open circles and lines show median and interquartile range; counts and proportions favoring Joint GP are displayed. This descriptive unique-pair summary differs from the equal-seed bootstrap estimand.')]
    for name,caption in plots:
        text += [r'\begin{figure}[p]\centering',r'\includegraphics[width=\textwidth,height=.84\textheight,keepaspectratio]{'+name+'.pdf}',r'\caption{'+caption+'}',r'\end{figure}',r'\clearpage']
    text += [r'\end{document}']
    (paper/'controlled-joint-ablation.tex').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
