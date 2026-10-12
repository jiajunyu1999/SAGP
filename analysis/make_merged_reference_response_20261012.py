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
def violin(ax,values,position,color,width=.7):
 values=np.asarray(values,float);assert np.isfinite(values).all() and np.ptp(values)>0
 v=ax.violinplot([values],positions=[position],widths=width,orientation='horizontal',showmeans=False,showmedians=False,showextrema=False,points=160,bw_method=.4)
 for body in v['bodies']:body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.45);body.set_linewidth(.65)
def main():
 AUDIT.mkdir(parents=True,exist_ok=True);m=pd.read_csv(EXP/'metrics_by_seed.csv');ci=pd.read_csv(EXP/'paired_component_intervals.csv');errors=pd.read_csv(EXP/'all_pair_errors.csv');rng=np.random.default_rng(17)
 fig,axes=plt.subplots(3,3,figsize=(10.6,10),gridspec_kw={'height_ratios':[1,1.15,1.3]});cats=[('nested',3),('nested',10),('nested',30),('nested',100),('shuffle_shift',100)];labels=['3 references','10 references','30 references','100 references','Shuffled response']
 for t in [1,2]:
  ax=axes[0,t-1]
  for i,(condition,budget) in enumerate(cats):
   y=m.query('task==@t and condition==@condition and budget==@budget').AP;col=RED if i==3 else (PEACH if i==4 else BLUE)
   assert len(y)==5;violin(ax,y,i,col)
   lo,med,hi=np.quantile(y,[.25,.5,.75]);ax.plot([lo,hi],[i,i],color=INK,lw=2.2,zorder=3);ax.scatter([med],[i],s=15,color='white',edgecolors=INK,linewidths=.7,zorder=5)
   ax.scatter(y,i+rng.uniform(-.14,.14,len(y)),color=col,s=17,edgecolors='white',linewidths=.4,alpha=.9,zorder=4)
  ax.set(yticks=range(5),yticklabels=labels,xlabel=f'Task {t} AP',ylim=(4.5,-.5));ax.text(.03,1.02,'Five splits; median / IQR',transform=ax.transAxes,fontsize=8)
 ax=axes[0,2];plot=[];bootrows=[]
 metrics=['AP','pair_macro_AUROC','macro_MAE']
 for t in [1,2]:
  bootstrap=np.load(EXP/f'bootstrap_task{t}_k100_minus_shuffle_shift_k100.npy');assert bootstrap.shape==(2001,3)
  for j,metric in enumerate(metrics):
   r=ci.query('task==@t and condition=="shuffle_shift" and metric==@metric').iloc[0];sign=-1 if metric=='macro_MAE' else 1
   values=sign*bootstrap[1:,j];v=sign*bootstrap[0,j];lo,hi=np.quantile(values,[.025,.975])
   assert np.isclose(v,sign*r.k100_minus_control) and np.allclose([lo,hi],sorted([sign*r.CI_low,sign*r.CI_high]))
   plot.append((t,metric,v,lo,hi,values));bootrows.extend({'task':t,'metric':metric,'draw':k+1,'benefit':float(value)} for k,value in enumerate(values))
 for i,(t,k,v,lo,hi,values) in enumerate(plot):
  col=RED if t==1 else BLUE;violin(ax,values,i,col,width=.7);ax.plot([lo,hi],[i,i],color=INK,lw=1.4,zorder=3);ax.scatter([v],[i],s=17,color=col,edgecolors='white',linewidths=.6,zorder=4)
 ax.axvline(0,ls='--',lw=.8,color='#A9B1BA');ax.set(yticks=range(6),yticklabels=[f'T{t} '+{'AP':'AP','pair_macro_AUROC':'pair AUROC','macro_MAE':'MAE reduction'}[k] for t,k,*_ in plot],ylim=(5.5,-.5),xlabel='Intact − shuffled: benefit');ax.text(.02,1.02,'2,000 paired bootstrap draws',transform=ax.transAxes,fontsize=8)
 pd.DataFrame(bootrows).to_csv(AUDIT/'figure5c_bootstrap_draws.csv',index=False)
 pairdata=[]
 for t in [1,2]:
  g=errors.query('task==@t and condition=="shuffle_shift"').groupby('pair_id')[['MAE_k100','MAE_control']].mean().reset_index();pair_scatter(axes[1,t-1],g,'MAE_control','MAE_k100','Shuffled-response MAE','Intact-response MAE',RED if t==1 else BLUE);g['task']=t;pairdata.append(g)
 ax=axes[1,2]
 for i,t in enumerate([1,2]):
  g=errors.query('task==@t and condition=="nested" and budget==3').groupby('pair_id')[['MAE_k100','MAE_control']].mean();gain=g.MAE_control-g.MAE_k100;ax.scatter(gain,i+rng.uniform(-.2,.2,len(gain)),s=11,color=RED if t==1 else BLUE,alpha=.5,rasterized=True);lo,med,hi=np.quantile(gain,[.25,.5,.75]);ax.errorbar(med,i,xerr=[[med-lo],[hi-med]],fmt='o',color=INK,ms=4,lw=1.5);ax.text(.02,.97-i*.075,f'T{t}: n={len(gain)}; {(gain>0).mean():.0%} lower error',transform=ax.transAxes,fontsize=8,va='top')
 ax.axvline(0,ls='--',lw=.8,color='#A9B1BA');ax.set(yticks=[0,1],yticklabels=['Task 1','Task 2'],ylim=(1.5,-.9),xlabel='MAE reduction: 3 → 100 references');ax.text(.02,1.02,'One dot per pair; median / IQR',transform=ax.transAxes,fontsize=8)
 source=BASE/'paper/baseline_table_sources_final_20261009/metrics_by_seed.csv'
 models=['fixed_tanimoto_gp','neural_mean_gp','global_monotone','conditional_monotone','context_neural']
 names=['Fixed kernel','Neural mean','Global mean','Conditional mean','SAGP']
 response=pd.read_csv(source);response=response[response.task.isin([1,2]) & response.model.isin(models)].copy()
 from matplotlib.patches import Patch
 from matplotlib.ticker import MaxNLocator
 for c,(metric,label) in enumerate([('AP','AP'),('pair_macro_AUROC','Within-pair AUROC'),('macro_MAE','Macro MAE (pIC50)')]):
  ax=axes[2,c];ax.axhspan(3.52,4.48,color='#fef9cf',alpha=.65,zorder=0)
  for y,model in enumerate(models):
   for task,offset,color in [(1,-.19,BLUE),(2,.19,PEACH)]:
    vals=response.loc[response.task.eq(task)&response.model.eq(model),metric].to_numpy();assert len(vals)==5 and np.isfinite(vals).all()
    ax.boxplot([vals],positions=[y+offset],widths=.29,orientation="horizontal",patch_artist=True,manage_ticks=False,showfliers=False,whis=1.5,
     boxprops={'facecolor':color,'edgecolor':color,'alpha':.65,'linewidth':.8},medianprops={'color':INK,'linewidth':1.2},
     whiskerprops={'color':color,'linewidth':.8},capprops={'color':color,'linewidth':.8})
    ax.scatter(vals,y+offset+rng.uniform(-.065,.065,5),s=13,color=color,edgecolors='white',linewidths=.35,zorder=4)
  ax.set(yticks=range(5),yticklabels=names,ylim=(4.6,-.6),xlabel=label)
  ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.xaxis.set_major_locator(MaxNLocator(5))
  ax.get_yticklabels()[-1].set_fontweight('bold')
  if c==0:ax.text(.02,1.02,'Five splits per task',transform=ax.transAxes,fontsize=8)
  if c==1:ax.legend(handles=[Patch(facecolor=BLUE,label='Task 1'),Patch(facecolor=PEACH,label='Task 2')],loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,fontsize=8,handlelength=1,columnspacing=1.2,borderaxespad=0)
  if c==2:ax.text(.02,1.02,'Median / IQR; all split points',transform=ax.transAxes,fontsize=8)
 D=FIG/'sagp_response_analysis_20261012';D.mkdir(exist_ok=True)
 response.to_csv(D/'response_metrics_by_seed.csv',index=False)
 (D/'manifest.json').write_text(json.dumps({'source':str(source.relative_to(BASE)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'models':models,'statistics':'Task 1 and Task 2 shown separately; five dependent scaffold evaluations per box; median, Q1-Q3, 1.5 IQR whiskers, all actual split points','figure':'Figure 5g-i','note':'Response formulations have different objectives; this is not an equal-information architecture ablation.'},indent=2))
 for a,l in zip(axes.flat,'abcdefghi'):panel(a,l)
 fig.tight_layout(h_pad=2.4,w_pad=2);save(fig,'figure5_reference_evidence','reference_interventions');pd.concat(pairdata).to_csv(AUDIT/'reference_pair_errors.csv',index=False);m.to_csv(AUDIT/'reference_metrics_by_seed.csv',index=False);ci.to_csv(AUDIT/'reference_intervals.csv',index=False)
 (AUDIT/'figure5_merged_manifest.json').write_text(json.dumps({'panels':'a-b split AP violins; c paired bootstrap benefit violins; d-f pair evidence; g-i separate-task response boxplots','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_synthetic_observations':True,'violin_a_b':'five actual dependent split observations; KDE, median and IQR','violin_c':'2000 existing paired component-bootstrap draws per task/metric, excluding row 0 original estimate; original estimate and percentile 95% interval overlaid; resamples are not independent experimental observations'},indent=2))
if __name__=='__main__':main()
