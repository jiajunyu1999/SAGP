"""Observed-data pipeline and nine-panel dataset audit; preserve Figures 3/4."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'outputs/datasets/sar_transfer_20261007'
OUT=BASE/'outputs/figures'
AUDIT=OUT/'figure2_two_tasks_20261012'
AUDIT.mkdir(exist_ok=True)
from publication_palette_20261009 import BLUE,RED,PEACH,SAND,YELLOW,MINT,LIGHT_BLUE,INK,GRAY,REFERENCE_PALETTE
TEAL,ORANGE,ROSE,PURPLE=PEACH,SAND,RED,LIGHT_BLUE
RNG=np.random.default_rng(20261009)
SOURCES={};SUMMARY={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.5,
    'xtick.labelsize':9.3,'ytick.labelsize':9.3,'axes.linewidth':.65,
    'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none',
    'axes.unicode_minus':False})


# Frozen numeric-mention parser, copied to keep plotting independent of trainers.
def numeric_assay(text):
    text=str(text)
    fields=[]
    units={"pm":1e-3,"nm":1,"um":1e3,"µm":1e3,"μm":1e3,"mm":1e6,"m":1e9}
    for name in ["atp","nad","nadh","nadph","dtt","bsa"]:
        pattern=rf"\b{name}\b[^.;]{{0,35}}?(\d+(?:\.\d+)?)\s*(pM|nM|uM|µM|μM|mM|M)\b"
        match=re.search(pattern,text,re.I)
        if match is None:
            pattern=rf"(\d+(?:\.\d+)?)\s*(pM|nM|uM|µM|μM|mM|M)\s*(?:of\s+)?{name}\b"
            match=re.search(pattern,text,re.I)
        value=float(match.group(1))*units[match.group(2).lower()] if match else np.nan
        fields.append(np.log10(value) if value>0 else np.nan)
    ph=re.search(r"\bpH\s*(?:of\s*|=\s*|:\s*)?(\d+(?:\.\d+)?)",text,re.I)
    fields.append(float(ph.group(1)) if ph else np.nan)
    duration=re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|h|min|s)\b",text,re.I)
    if duration:
        unit=duration.group(2).lower()
        mult=60 if unit.startswith("h") else 1/60 if unit.startswith("s") else 1
        fields.append(np.log1p(float(duration.group(1))*mult))
    else:fields.append(np.nan)
    temp=re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*|degrees?\s*|deg\.?\s*)C\b",text,re.I)
    fields.append(float(temp.group(1)) if temp else np.nan)
    return np.asarray(fields,dtype=np.float32)

def read(relative):
    p=BASE/relative;SOURCES[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text()) if p.suffix=='.json' else pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def clean(ax,xgrid=False):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.set_axisbelow(True);ax.tick_params(length=3,width=.65,pad=3)
    if xgrid:ax.grid(axis='x',ls='--',lw=.5,alpha=.65,color=GRAY,zorder=0)


def letter(ax,key):
    ax.text(-.06,1.06,key,transform=ax.transAxes,fontsize=13,fontweight='bold',color=INK)


def card(ax,xy,wh,text,color,fontsize=10):
    x,y=xy;w,h=wh
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',
        facecolor={BLUE:LIGHT_BLUE,TEAL:MINT,ORANGE:YELLOW,ROSE:SAND,PURPLE:MINT}.get(color,'white'),edgecolor=color,linewidth=1,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color=INK,zorder=3)


def arrow(ax,a,b,color='#9BA6AF'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.9,color=color,zorder=1))


def cloud(ax,values,y,color,size=6,alpha=.25,jitter=.2):
    v=np.asarray(values,dtype=float);assert np.isfinite(v).all()
    ax.scatter(v,y+RNG.uniform(-jitter,jitter,len(v)),s=size,alpha=alpha,color=color,
        linewidths=0,rasterized=True,zorder=3)
    q1,med,q3=np.quantile(v,[.25,.5,.75])
    ax.plot([q1,q3],[y+.30,y+.30],color=color,lw=1.4,zorder=4)
    ax.scatter([med],[y+.30],s=22,facecolor='white',edgecolor=color,linewidth=.95,zorder=5)
    return med


def figure2(obs,sar,raw,conditions,contexts,test):
    fig=plt.figure(figsize=(10,11.1))
    grid=fig.add_gridspec(4,3,left=.10,right=.975,bottom=.055,top=.94,
        height_ratios=[1.9,1.9,1.65,1.65],hspace=.72,wspace=.85)
    axes=[fig.add_subplot(grid[r,c]) for r in range(2) for c in range(3)]
    for a,l in zip(axes,'abcdef'):clean(a);letter(a,l)
    # a: complete resource and the filtered repeated-pair pool use identical units.
    raw_support=raw.groupby('pair_id').size();filtered=obs.groupby('pair_id').size()
    survival=[];a=axes[0]
    for name,counts,color in [('Complete IC50',raw_support,BLUE),('Filtered repeats',filtered,TEAL)]:
        thresholds=np.sort(counts.unique());n=np.array([(counts>=v).sum() for v in thresholds])
        a.plot(thresholds,n,lw=1.2,color=color,label=name,zorder=2)
        a.scatter(thresholds,n,s=8,color=color,alpha=.65,linewidths=0,zorder=3)
        survival.extend(dict(cohort=name,shared_molecules=int(v),pairs=int(k)) for v,k in zip(thresholds,n))
    a.set_xscale('log');a.set_yscale('log');a.set_xlabel('Shared molecules / pair');a.set_ylabel('Pairs with ≥x molecules')
    a.set_xticks([1,10,100],['1','10','100']);a.legend(frameon=False,fontsize=8.1,loc='upper right',handlelength=1.2)
    a.text(.04,.12,f'{(raw_support==1).mean():.0%} singleton pairs',transform=a.transAxes,fontsize=9)
    pd.DataFrame(survival).to_csv(AUDIT/'figure2_support_survival.csv',index=False)
    # b: all matched molecules, organized by source relationship.
    a=axes[1];clean(a,True);a.spines['left'].set_visible(False)
    labels=[]
    for y,name,color in [(1,'same_document',BLUE),(0,'cross_document',PURPLE)]:
        h=obs[obs.source_type.eq(name)];med=cloud(a,h.abs_delta,y,color,size=2.5,alpha=.12)
        labels.append(f'{"Same doc." if y else "Cross doc."}\nn={len(h):,}; m={med:.2f}')
    a.set_yticks([1,0],labels,fontsize=8.7);a.tick_params(axis='y',length=0);a.set_ylim(-.45,1.45)
    a.set_xscale('symlog',linthresh=.15);a.set_xticks([0,1,3],[0,1,3]);a.set_xlabel('|B − A| (pIC50)')
    obs[['sample_id','pair_id','source_type','abs_delta']].to_csv(AUDIT/'figure2_source_disagreement.csv',index=False)
    # c: separate shared offset and within-pair molecular dispersion.
    pair=obs.groupby('pair_id').agg(shift=('signed_delta','median'),molecules=('sample_id','size'))
    deviations=abs(obs.signed_delta-obs.pair_id.map(pair['shift']))
    pair['MAD']=deviations.groupby(obs.pair_id).median()
    a=axes[2];a.scatter(pair['shift'],pair.MAD,s=5,color=BLUE,alpha=.15,linewidths=0,rasterized=True,zorder=3)
    a.set_xlabel('Pair median shift (pIC50)');a.set_ylabel('Within-pair MAD (pIC50)')
    a.set_xscale('symlog',linthresh=.25);a.set_yscale('symlog',linthresh=.1)
    a.set_xticks([-3,0,3],['−3','0','3']);a.set_yticks([0,.1,.5,2],[0,.1,.5,2])
    a.text(.10,1.025,f'{len(pair):,} pairs\nMedian MAD={pair.MAD.median():.2f}',transform=a.transAxes,va='bottom',fontsize=9)
    pair.reset_index().to_csv(AUDIT/'figure2_pair_shift_dispersion.csv',index=False)
    # d: outcome-labelled questions, with A-only eligibility.
    questions=sar[['comparison_id','pair_id','target_chembl_id','source_type','anchor_gap_abs','gap_A','gap_B','outcome','binary_loss','tanimoto']].copy()
    questions['oriented_B']=np.sign(questions.gap_A)*questions.gap_B
    a=axes[3]
    for key,color in [('retained',BLUE),('reversed',ROSE),('attenuated_or_uncertain',ORANGE)]:
        h=questions[questions.outcome.eq(key)]
        a.scatter(h.anchor_gap_abs,h.oriented_B,s=2.3,color=color,alpha=.11,linewidths=0,rasterized=True,zorder=3)
    hi=max(questions.anchor_gap_abs.max(),questions.oriented_B.max())*1.04
    a.plot([0,hi],[0,hi],color='#9BA6AF',ls='--',lw=.75,zorder=1);a.axhline(0,lw=.7,color=GRAY,zorder=1)
    a.set_xlabel('Assay-A gap (pIC50)');a.set_ylabel('A-oriented B gap (pIC50)')
    a.text(.10,1.025,f'n={len(sar):,}; loss={sar.binary_loss.mean():.1%}',transform=a.transAxes,va='bottom',fontsize=9)
    questions.to_csv(AUDIT/'figure2_analogue_questions.csv',index=False)
    # e: support and event-rate heterogeneity; no outcome-based pair filtering.
    support=sar.groupby(['pair_id','source_type']).binary_loss.agg(['size','mean']).reset_index().rename(columns={'size':'comparisons','mean':'loss_rate'})
    a=axes[4]
    for name,color in [('same_document',BLUE),('cross_document',PURPLE)]:
        h=support[support.source_type.eq(name)]
        a.scatter(h.comparisons,h.loss_rate,s=8,color=color,alpha=.35,linewidths=0,rasterized=True,zorder=3)
    a.set_xscale('log');a.set_xlabel('Analogue comparisons / pair');a.set_ylabel('SAR-loss fraction')
    a.set_ylim(-.04,1.04);a.set_xticks([1,10,100,1000],['1','10','100','1k'])
    a.text(.10,1.045,f'{len(support):,} pairs',transform=a.transAxes,va='bottom',fontsize=9)
    support.to_csv(AUDIT/'figure2_support_event_rate.csv',index=False)
    # f: extraction coverage is not the true completeness of experimental protocols.
    needed=set(sar.assay_A)|set(sar.assay_B)
    ctx=contexts.drop_duplicates('assay_chembl_id').set_index('assay_chembl_id').loc[sorted(needed)]
    parsed=np.stack([numeric_assay(t) for t in ctx.assay_description.fillna('')]);present=np.isfinite(parsed).sum(0)
    fields=['ATP','NAD','NADH','NADPH','DTT','BSA','pH','Time','Temp.'];percent=100*present/len(ctx)
    a=axes[5];a.spines['left'].set_visible(False)
    for y,x in enumerate(percent):
        a.plot([0,x],[y,y],lw=1.2,color=BLUE,zorder=2);a.scatter([x],[y],s=23,color=BLUE,zorder=3)
        a.text(max(percent)*1.04,y,str(present[y]),va='center',fontsize=8.5)
    a.set_yticks(range(len(fields)),fields,fontsize=9);a.tick_params(axis='y',length=0)
    a.set_ylim(len(fields)-.5,-.8);a.set_xlim(0,max(percent)*1.26)
    a.set_xlabel('Numeric mention parsed (%)')
    a.text(.10,1.05,f'{len(ctx):,} assay descriptions',transform=a.transAxes,fontsize=8.8)
    pd.DataFrame({'field':fields,'parsed':present,'total':len(ctx),'percent':percent}).to_csv(AUDIT/'figure2_metadata_coverage.csv',index=False)
    # g: all observations in the separate condition audit.
    a=fig.add_axes([.205,.055,.35,.398]);clean(a,True);a.spines['left'].set_visible(False)
    order=['concentration_atp','concentration_nad','concentration_substrate','concentration_enzyme','concentration_dtt','ph','preincubation_time','reported_duration','percentage_albumin','percentage_serum','percentage_detergent','presence_absence','cell_line','readout_method']
    labels=['ATP','NAD','Substrate','Enzyme','DTT','pH','Preincubation','Reaction / treatment','Albumin (%)','Serum (%)','Detergent (%)','Additive ±','Cell line / host','Readout']
    colors=[BLUE]*6+[ORANGE]*2+[TEAL]*4+[PURPLE,ROSE]
    stats=[]
    for i,(key,color) in enumerate(zip(order,colors)):
        h=conditions[conditions.condition.eq(key)];y=13-i
        med=cloud(a,h.fold_difference,y,color,size=5.4,alpha=.65,jitter=.19)
        a.text(1.04,y,f'{len(h):,} | {h.pair_id.nunique():,} | {med:.1f}×',transform=a.get_yaxis_transform(),va='center',fontsize=8.7)
        stats.append(dict(condition=key,rows=len(h),pairs=h.pair_id.nunique(),median_fold=med))
    a.set_yticks(range(13,-1,-1),labels,fontsize=9.2);a.tick_params(axis='y',length=0)
    a.set_xscale('log');a.set_xlim(.8,conditions.fold_difference.max()*1.15);a.set_ylim(-.5,13.6)
    a.set_xticks([1,10,100,1000,10000],['1×','10×','100×','1k×','10k×']);a.set_xlabel('IC50 fold difference')
    for x in [10,100]:a.axvline(x,ls='--',color='#9BA6AF',lw=.7,zorder=1)
    fig.text(.03,.473,'g',fontsize=13,fontweight='bold',color=INK)
    a.text(1.04,1.015,'n | pairs | median',transform=a.transAxes,fontsize=8.6)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for l,c in [('Biochem.',BLUE),('Time',ORANGE),('Media',TEAL),('Cell',PURPLE),('Readout',ROSE)]],
        loc='center',bbox_to_anchor=(.405,.474),ncol=5,frameon=False,fontsize=8.5,handlelength=.7,columnspacing=.9)
    conditions.to_csv(AUDIT/'figure2_condition_observations.csv',index=False);pd.DataFrame(stats).to_csv(AUDIT/'figure2_condition_statistics.csv',index=False)
    # h: chemical-series/target concentration of the analogue cohort.
    a=fig.add_subplot(grid[2,2]);clean(a);letter(a,'h')
    targets=sar.groupby('target_chembl_id').size().sort_values(ascending=False).rename('comparisons').reset_index()
    targets['rank']=np.arange(1,len(targets)+1);targets['covered']=targets.comparisons.cumsum()/len(sar)
    a.plot(100*targets['rank']/len(targets),100*targets.covered,lw=.8,color=BLUE,zorder=2)
    a.scatter(100*targets['rank']/len(targets),100*targets.covered,s=5,color=BLUE,alpha=.5,linewidths=0,zorder=3)
    a.plot([0,100],[0,100],ls='--',lw=.7,color='#9BA6AF',zorder=1)
    a.set(xlim=(0,100),ylim=(0,103));a.set_xlabel('Targets retained (%)');a.set_ylabel('Comparisons covered (%)')
    top10=float(targets.iloc[:10].comparisons.sum()/len(sar))
    a.text(.10,1.035,f'{len(targets)} targets\nTop 10: {top10:.0%} of queries',transform=a.transAxes,va='bottom',fontsize=9)
    targets.to_csv(AUDIT/'figure2_target_coverage.csv',index=False)
    # i: five original tests; dots are seeds, not biological replicates.
    a=fig.add_subplot(grid[3,2]);clean(a,True);letter(a,'i');a.spines['left'].set_visible(False)
    selected=test[test.task.isin([1,2]) & test.model.eq('joint_monotone')].copy()
    selected['prevalence']=selected.positive/selected.n
    for task in [1,2]:
        h=selected[selected.task.eq(task)];assert len(h)==5
        y=3-task;a.scatter(100*h.prevalence,y+RNG.uniform(-.07,.07,5),s=30,color=[BLUE,TEAL,ORANGE][task-1],alpha=.7,linewidths=0,zorder=3)
        a.text(.02,y+.28,f'n={h.n.min():,}–{h.n.max():,}',transform=a.get_yaxis_transform(),fontsize=8.4,va='center')
    a.set_yticks([2,1],['Task 1','Task 2']);a.tick_params(axis='y',length=0)
    a.set_ylim(.5,2.7);a.set_xlim(0,36);a.set_xlabel('Test positive fraction (%)')
    selected.to_csv(AUDIT/'figure2_test_prevalence.csv',index=False)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for l,c in [('Retained',BLUE),('Reversed',ROSE),('Attenuated / uncertain',ORANGE)]],
        loc='upper center',bbox_to_anchor=(.51,.997),ncol=3,frameon=False,fontsize=9.1,handlelength=1,columnspacing=1.8)
    SUMMARY['figure2']={'panels':9,'paired_observations':len(obs),'pair_shift_dispersion_points':len(pair),
        'analogue_comparisons':len(sar),'analogue_pairs':len(support),'assay_descriptions':len(ctx),
        'condition_rows':len(conditions),'condition_types':len(stats),'targets':len(targets),
        'top10_target_fraction':top10,'singleton_pair_fraction':float((raw_support==1).mean())}
    save(fig,'figure2_data_audit')


def main():
    before={n:hashlib.sha256((OUT/f'{n}.png').read_bytes()).hexdigest() for n in ['figure3_model_gains','figure4_casebook']}
    obs=read('outputs/datasets/sar_transfer_20261007/molecule_assay_observations.parquet')
    sar=read('outputs/datasets/sar_transfer_20261007/anchor_transfer_all.parquet')
    raw=read('data/processed/pairs.parquet');raw=raw[raw.endpoint.eq('IC50')] if 'endpoint' in raw else raw
    conditions=read('outputs/figures/condition_scatter_20261007/scatter_plot_data.csv')
    contexts=read('outputs/datasets/sar_transfer_20261007/assay_contexts.parquet')
    test=read('paper/baseline_table_sources_final_20261009/metrics_by_seed_and_draw.csv')
    counts=read('outputs/audit/curation_counts.json')
    assert counts['candidate_rows']==1563556 and counts['endpoints']['IC50']==690683
    assert len(raw)==153468 and raw.pair_id.nunique()==83381
    assert len(obs)==79810 and obs.pair_id.nunique()==12392
    assert len(sar)==73950 and len(conditions)==4626
    figure2(obs,sar,raw,conditions,contexts,test)
    SUMMARY.update(palette=REFERENCE_PALETTE,palette_source="paper/image.png",
        palette_script_sha256=hashlib.sha256((BASE/'analysis/publication_palette_20261009.py').read_bytes()).hexdigest(),source_sha256=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numeric_parser_script_sha256=hashlib.sha256((BASE/'analysis/train_sar_transfer_baseline_20261007.py').read_bytes()).hexdigest(),
        other_figures_at_start=before,jitter_seed=20261009,
        units='Activity records, molecular observations, analogue comparisons, assay pairs and sample-condition rows differ.',
        interpretation='All observations displayed; conditions can covary; metadata coverage means parsed numeric mentions, not true protocol availability.',
        empirical_selection='Largest shared-molecule pair, without inspecting model performance or outcomes.')
    (AUDIT/'manifest.json').write_text(json.dumps(SUMMARY,indent=2))
    (AUDIT/'README.txt').write_text('Run: python analysis/make_information_dense_figures12_20261009.py\nFigure1: source/curation/model pipeline, then largest-supported-pair measured examples; these are descriptive data, not fitted predictions.\nFigure2: complete/filtered pair-support survival; source-stratified disagreement; pair offset/MAD; all analogue gaps; pair support/loss rate; numeric mention coverage; separate 14-condition audit; target coverage; five test prevalences.\nAll original observations plotted without downsampling or sign filtering.\nQuartile intervals describe observations, not confidence intervals.\nCondition cohorts overlap and do not establish causation.\nThis script writes Figures1/2 only.\n')
    print(json.dumps({k:v for k,v in SUMMARY.items() if k in ['figure1','figure2']},indent=2))


if __name__=='__main__':main()
