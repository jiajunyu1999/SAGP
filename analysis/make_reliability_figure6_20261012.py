"""Dense real-query/pair evidence; no synthetic observations or sparse trend lines."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_palette_20261009 import RED,PEACH,BLUE,LIGHT_BLUE,GRAY,INK
BASE=Path(__file__).resolve().parents[1];DATA=BASE/'outputs/datasets/sar_transfer_20261007/additional_frozen_analyses_20261009';FIG=BASE/'outputs/figures';AUDIT=FIG/'dense_additional_20261009';EXP=DATA/'reference_interventions'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.edgecolor':INK,'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False,'svg.fonttype':'none'})
def panel(ax,letter):ax.text(-.06,1.08,letter,transform=ax.transAxes,fontsize=13,fontweight='bold');ax.tick_params(labelsize=8)
def pair_scatter(ax,g,x,y,xlabel,ylabel,col):
 ax.scatter(g[x],g[y],s=12,color=col,alpha=.55,edgecolors='none',rasterized=True,zorder=3)
 high=max(g[[x,y]].max().max(),.1)*1.08;ax.plot([0,high],[0,high],ls='--',color='#A9B1BA',lw=.8,zorder=0);ax.set(xscale='symlog',yscale='symlog',xlabel=xlabel,ylabel=ylabel,xlim=(0,high),ylim=(0,high));ax.set_xscale('symlog',linthresh=.05);ax.set_yscale('symlog',linthresh=.05)
 ticks=[v for v in [0,.1,.3,1,3,10] if v<=high]
 ax.set_xticks(ticks,labels=[str(v) for v in ticks]);ax.set_yticks(ticks,labels=[str(v) for v in ticks]);ax.text(.03,.97,f'n={len(g):,} pairs\n{(g[y]<g[x]).mean():.0%} lower error',transform=ax.transAxes,va='top',fontsize=8)
def save(fig,name,alias):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 for ext in ['pdf','png','svg']:shutil.copy2(FIG/(name+'.'+ext),DATA/(alias+'.'+ext))
def main():
 AUDIT.mkdir(parents=True,exist_ok=True)
 q=pd.read_csv(DATA/'query_support_audit.csv');q['absolute_error']=abs(q.target-q.regression);cal=pd.read_csv(DATA/'calibration_by_seed.csv');fig,axes=plt.subplots(2,2,figsize=(9.8,6.4));retained=[];paircal=[]
 for t in [1,2]:
  key='comparison_id' if t==1 else 'sample_id';f=q[q.task.eq(t)].copy()
  rows=[]
  for (seed,pair),h in f.groupby(['seed','pair_id'],sort=False):
   k=max(1,int(np.ceil(.2*len(h))));selected=h.sort_values('variance',kind='stable').iloc[:k];rows.append(dict(task=t,seed=seed,pair_id=pair,full_MAE=h.absolute_error.mean(),retained_MAE=selected.absolute_error.mean(),n=len(h),retained_n=k))
  rr=pd.DataFrame(rows);gg=rr.groupby('pair_id')[['full_MAE','retained_MAE']].mean().reset_index();pair_scatter(axes[0,t-1],gg,'full_MAE','retained_MAE','All-query pair MAE','20% retained pair MAE',RED if t==1 else BLUE);retained.append(rr)
  pc=f.groupby(['seed','pair_id']).agg(mean_risk=('prediction','mean'),event_rate=('binary_loss','mean'),n=(key,'size')).reset_index();pc=pc[pc.n>=10];pp=pc.groupby('pair_id')[['mean_risk','event_rate','n']].mean().reset_index();ax=axes[1,t-1];ax.scatter(pp.mean_risk,pp.event_rate,s=12,color=BLUE if t==1 else PEACH,alpha=.55,edgecolors='none',rasterized=True);ax.plot([0,1],[0,1],ls='--',color='#A9B1BA',lw=.8,zorder=0);ax.set(xlabel='Mean predicted risk per pair',ylabel='Observed event rate per pair',xlim=(-.02,1.02),ylim=(-.02,1.02));cc=cal.query('task==@t and model=="context_neural"');ax.text(.03,.97,f'n={len(pp):,} pairs\nQuery ECE={cc.ECE10.mean():.3f}',transform=ax.transAxes,va='top',fontsize=9);pc['task']=t;paircal.append(pc)
 for a,l in zip(axes.flat,'abcd'):panel(a,l)
 for ax in axes.flat:
  for annotation in ax.texts:
   if annotation.get_fontsize()<12:annotation.set_bbox(dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.2))
 fig.tight_layout(h_pad=2.5,w_pad=2);save(fig,'figure6_applicability_reliability','additional_diagnostics');pd.concat(retained).to_csv(AUDIT/'pair_selective_errors_by_seed.csv',index=False);pd.concat(paircal).to_csv(AUDIT/'pair_reliability_by_seed.csv',index=False)
 (AUDIT/'manifest.json').write_text(json.dumps({'data':'actual query/assay-pair predictions; no pseudo-replicates','reference_points':'five dependent scaffold evaluations','pair_points':'average available split-specific metrics for each unique pair','panels':'Figure 6 a-d selective error and reliability','intervals':'2000 paired assay/document-component bootstrap draws; fixed-model conditional','selection':'20%ceil per pair by variance; every pair represented','source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'query_support_audit.csv',EXP/'metrics_by_seed.csv',EXP/'all_pair_errors.csv',EXP/'paired_component_intervals.csv',Path(__file__)]}},indent=2))
if __name__=='__main__':main()
