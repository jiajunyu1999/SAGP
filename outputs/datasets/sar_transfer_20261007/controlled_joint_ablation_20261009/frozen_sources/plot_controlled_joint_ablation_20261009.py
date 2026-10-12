"""Full controlled ablation, with repeated-seed and paired-interval evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from controlled_joint_ablation_20261009 import RESULTS,MODELS
from publication_palette_20261009 import BLUE,RED,PEACH,SAND,YELLOW,INK,GRAY

LABELS={'task1_only':'A  Task 1 only','task2_only':'B  Task 2 only','joint':'C  Joint GP',
        'no_molecular_residual':'D  No chemical residual','no_monotone_mean':'E  Nonmonotone mean',
        'no_contrast_covariance':'F  Independent variance','shared_multihead':'G  Shared, separate heads'}
KEYS=['AP','pair_macro_AUROC','macro_MAE']
NAMES=['Average precision','Within-pair AUROC','Macro MAE (pIC50)']
OUT=RESULTS/'figures'


def clean(ax,letter):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.tick_params(length=3,width=.65)
    ax.grid(axis='x',color=GRAY,ls='--',lw=.45,alpha=.5,zorder=0)
    ax.text(-.06,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,color=INK)


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':10,'axes.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
    metrics=pd.read_csv(RESULTS/'metrics_by_seed.csv').query('partition=="test"')
    ci=pd.read_csv(RESULTS/'paired_component_bootstrap.csv')
    fig,axes=plt.subplots(2,3,figsize=(11.7,7.1));fig.subplots_adjust(left=.24,right=.98,bottom=.08,top=.90,wspace=.22,hspace=.35)
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Validation selected: five seeds'),Line2D([],[],color=PEACH,marker='o',label='Final budget: five seeds'),Line2D([],[],color=INK,marker='o',mfc='white',label='Mean ± seed SD')],loc='upper center',ncol=3,frameon=False)
    for t in [1,2]:
        for j,key in enumerate(KEYS):
            ax=axes[t-1,j];clean(ax,chr(ord('a')+(t-1)*3+j))
            for y,model in enumerate(MODELS):
                if model=='joint':ax.axhspan(y-.46,y+.46,color=YELLOW,zorder=0)
                for ck,offset,color in [('selected',-.13,BLUE),('final',.13,PEACH)]:
                    vals=metrics.query('task==@t and model==@model and checkpoint==@ck').sort_values('seed')[key].to_numpy()
                    assert len(vals)==5
                    ax.scatter(vals,y+offset+np.linspace(-.045,.045,5),s=14,color=color,alpha=.75,linewidths=0,zorder=3)
                    ax.errorbar(vals.mean(),y+offset,xerr=vals.std(ddof=1),fmt='o',ms=4,mfc='white',mec=color,ecolor=color,lw=1.1,zorder=4)
            ax.set_yticks(range(len(MODELS)),[LABELS[m] for m in MODELS] if j==0 else ['']*len(MODELS));ax.set_ylim(6.7,-.7)
            ax.tick_params(axis='y',length=0);ax.set_xlabel(NAMES[j]);ax.text(.02,1.04,f'Task {t}',transform=ax.transAxes,fontsize=10,color=INK)
    save(fig,'controlled_ablation_seed_comparison')
    controls=[m for m in MODELS if m!='joint']
    fig,axes=plt.subplots(2,3,figsize=(11.7,7.1));fig.subplots_adjust(left=.24,right=.98,bottom=.08,top=.90,wspace=.30,hspace=.35)
    fig.legend(handles=[Line2D([],[],color=RED,marker='o',label='Joint − control: selected'),Line2D([],[],color=BLUE,marker='o',label='Joint − control: final'),Line2D([],[],color=INK,label='95% component bootstrap interval')],loc='upper center',ncol=3,frameon=False)
    for t in [1,2]:
        for j,key in enumerate(KEYS):
            ax=axes[t-1,j];clean(ax,chr(ord('a')+(t-1)*3+j));ax.axvline(0,color=INK,ls='--',lw=.7,zorder=1)
            for y,model in enumerate(controls):
                for ck,offset,color in [('selected',-.14,RED),('final',.14,BLUE)]:
                    r=ci.query('task==@t and control==@model and metric==@key and checkpoint==@ck').iloc[0]
                    ax.plot([r.CI_low,r.CI_high],[y+offset,y+offset],color=color,lw=1.2,zorder=3)
                    ax.scatter(r.joint_minus_control,y+offset,s=20,color=color,linewidths=0,zorder=4)
            ax.set_yticks(range(6),[LABELS[m] for m in controls] if j==0 else ['']*6);ax.set_ylim(5.7,-.7);ax.tick_params(axis='y',length=0)
            ax.set_xlabel('Δ '+NAMES[j]);ax.text(.02,1.04,f'Task {t}; '+('lower is better' if j==2 else 'higher is better'),transform=ax.transAxes,fontsize=9.2,color=INK)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    save(fig,'controlled_ablation_paired_intervals')
    pairs=pd.read_csv(RESULTS/'paired_assay_pair_differences.csv').query('checkpoint=="selected"')
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.8));fig.subplots_adjust(left=.24,right=.92,bottom=.09,top=.96,wspace=.55,hspace=.38)
    rng=np.random.default_rng(20261009)
    for t in [1,2]:
        for j,key in enumerate(['pair_macro_AUROC','macro_MAE']):
            ax=axes[t-1,j];clean(ax,chr(ord('a')+(t-1)*2+j));ax.axvline(0,color=INK,ls='--',lw=.7,zorder=1)
            for y,model in enumerate(controls):
                group=pairs.query('task==@t and model==@model').groupby('pair_id')[key+'_joint_minus_control'].mean().dropna()
                v=group.to_numpy();color=RED if model in ['task1_only','task2_only','shared_multihead'] else BLUE
                ax.scatter(v,y+rng.uniform(-.18,.18,len(v)),s=9,color=color,alpha=.45,linewidths=0,rasterized=True,zorder=3)
                q1,med,q3=np.quantile(v,[.25,.5,.75]);ax.plot([q1,q3],[y+.29,y+.29],color=color,lw=1.4,zorder=4)
                ax.scatter(med,y+.29,s=19,facecolor='white',edgecolor=color,linewidth=.9,zorder=5)
                better=(v<0).mean() if key=='macro_MAE' else (v>0).mean()
                ax.text(1.025,y,f'{len(v)} | {better:.0%}',transform=ax.get_yaxis_transform(),va='center',fontsize=8.7)
            ax.set_yticks(range(6),[LABELS[m] for m in controls] if j==0 else ['']*6);ax.set_ylim(5.7,-.6);ax.tick_params(axis='y',length=0)
            ax.set_xlabel('Joint − control: '+('AUROC' if j==0 else 'MAE (pIC50)'))
            ax.text(.02,1.04,f'Task {t}',transform=ax.transAxes,fontsize=10,color=INK)
            ax.text(1.025,1.015,'pairs | better',transform=ax.transAxes,fontsize=8.7)
    save(fig,'controlled_ablation_unique_pair_changes')


if __name__=='__main__':main()
