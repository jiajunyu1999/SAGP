"""Reference-matched three-tier Fig1 with vector cards/icons and actual-data plots."""
from pathlib import Path
import shutil,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Ellipse,Polygon,Rectangle
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdDepictor
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'outputs/figures';AUDIT=OUT/'figure1_two_tasks_20261012';AUDIT.mkdir(exist_ok=True)
NAVY='#15355A';BLUE='#649cc9';LIGHT='#a3cbe0';RED='#e1585d';PEACH='#f29f7b';SAND='#fbd49f';MINT='#d6eae3';GREEN='#76AD9E';GRAY='#AAB4BE';PURPLE='#AEA7CD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8})
TEXTS=[]
MOL_BANK={}
MOL_IDS=[]
MOL_CHANGED={}
def text(x,y,s,size=13,weight='normal',ha='center',va='center',color=NAVY,box=None):
 t=ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va=va,color=color,zorder=10,linespacing=1.15)
 if box:TEXTS.append((t,box))
 return t
def card(x,y,w,h,color=BLUE,heading=None,hs=14):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.55',edgecolor=color,facecolor=tuple(.9+.1*np.array(to_rgb(color))),linewidth=1.15,zorder=4);ax.add_patch(p)

 if heading:text(x+w/2,y+h-1.7,heading,hs,'bold',box=(x+.3,y+h-3.15,x+w-.3,y+h-.2))
 return p
def arrow(a,b,col=BLUE,lw=2.1,scale=19):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,color=col,lw=lw,zorder=5,shrinkA=0,shrinkB=0))
def graph(cx,cy,size=2.3,col=BLUE,variant=0,highlight=False):
    """Flat single-colour shape icon preserving a real molecule's connectivity."""
    ident=MOL_IDS[int(variant)%len(MOL_IDS)]
    mol=Chem.MolFromSmiles(MOL_BANK[ident])
    if mol is None:raise ValueError(f'Invalid canonical SMILES for {ident}')
    rdDepictor.Compute2DCoords(mol)
    conf=mol.GetConformer()
    raw=np.array([[conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    raw[:,1]*=-1
    width=4.3*size;height=2.4*size
    span=np.maximum(np.ptp(raw,axis=0),1e-6)
    scale=min((width-.4*size)/span[0],(height-.4*size)/span[1])
    pts=(raw-(raw.max(axis=0)+raw.min(axis=0))/2)*scale+[cx,cy]
    distances=[np.linalg.norm(pts[b.GetBeginAtomIdx()]-pts[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    radius=min(.14*size,min(distances)*.22)
    for bond in mol.GetBonds():
        p0,p1=pts[bond.GetBeginAtomIdx()],pts[bond.GetEndAtomIdx()]
        ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=col,lw=1.8,zorder=7,solid_capstyle='round')
    for x,y in pts:
        ax.add_patch(Circle((x,y),radius,facecolor=col,edgecolor='none',zorder=8))

def document(x,y,w=2.3,h=3.5):
 ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h-.65),(x+w-.65,y+h),(x,y+h)],closed=True,facecolor='white',edgecolor=BLUE,lw=1.5,zorder=6))
 for yy in np.linspace(y+.65,y+h-1,3):ax.plot([x+.35,x+w-.4],[yy,yy],color=BLUE,lw=1,zorder=7)
def curve(x,y,w,h,mode='mean',color=BLUE):
 a=fig.add_axes([x/100,y/75,w/100,h/75]);u=np.linspace(0,1,80)
 v=.2+.65/(1+np.exp(-9*(u-.45))) if mode=='mean' else (.13+.42*np.exp(-((u-.45)/.24)**2))
 sd=.07+.18*np.exp(-((u-.45)/.3)**2);a.fill_between(u,v-sd,v+sd,color=LIGHT,alpha=.45);a.plot(u,v,color=color,lw=1.6);a.set(xlim=(0,1),ylim=(0,1));a.set_xticks([]);a.set_yticks([]);a.spines['bottom'].set_color(GRAY);a.spines['left'].set_color(GRAY)
 return a

def main():
 global fig,ax,MOL_IDS,MOL_BANK,MOL_CHANGED
 obs=pd.read_parquet(BASE/'outputs/datasets/sar_transfer_20261007/molecule_assay_observations.parquet');sar=pd.read_parquet(BASE/'outputs/datasets/sar_transfer_20261007/anchor_transfer_all.parquet');raw=pd.read_parquet(BASE/'data/processed/pairs.parquet');raw=raw[raw.endpoint.eq('IC50')] if 'endpoint' in raw else raw
 chosen=obs.groupby('pair_id').size().idxmax();g=obs[obs.pair_id.eq(chosen)];q=sar[sar.pair_id.eq(chosen)].copy();q['oriented_B']=np.sign(q.gap_A)*q.gap_B
 molecules=pd.read_csv(BASE/'data/processed/molecules.csv',usecols=['chemical_identity','canonical_isomeric_smiles']).drop_duplicates('chemical_identity').set_index('chemical_identity');bank=molecules.canonical_isomeric_smiles.to_dict()
 # Use one concise, actual same-target analogue pair for the molecular
 # drawings; retain the largest supported pair above for the empirical panels.
 illustrative_pair_id='pair_1239ea7f2b52b1c322ad87c1'
 illustrative_ids=['id_3d3334fc19c43e5d234a8ae9','id_a8a6ef290c8fb84dd3f615b6']
 pair_rows=sar[sar.pair_id.eq(illustrative_pair_id)]
 pair=pair_rows[((pair_rows.identity_1.eq(illustrative_ids[0]))&(pair_rows.identity_2.eq(illustrative_ids[1])))|((pair_rows.identity_1.eq(illustrative_ids[1]))&(pair_rows.identity_2.eq(illustrative_ids[0])))].iloc[0]
 MOL_IDS=illustrative_ids.copy()
 # Add distinct examples from this same assay pair for the reference/query icons.
 display_obs=obs[obs.pair_id.eq(illustrative_pair_id)]
 extra=[i for i in display_obs.chemical_identity.drop_duplicates() if i not in MOL_IDS and i in bank]
 extra=sorted(extra,key=lambda i:(Chem.MolFromSmiles(bank[i]).GetNumHeavyAtoms(),i))[:4]
 MOL_IDS += extra;MOL_BANK=bank;assert all(Chem.MolFromSmiles(bank[i]) is not None for i in MOL_IDS)
 mcs=rdFMCS.FindMCS([Chem.MolFromSmiles(bank[MOL_IDS[0]]),Chem.MolFromSmiles(bank[MOL_IDS[1]])],ringMatchesRingOnly=True,completeRingsOnly=True,timeout=5)
 query=Chem.MolFromSmarts(mcs.smartsString);MOL_CHANGED={k:[i for i in range(Chem.MolFromSmiles(bank[MOL_IDS[k]]).GetNumAtoms()) if i not in set(Chem.MolFromSmiles(bank[MOL_IDS[k]]).GetSubstructMatch(query))] for k in (0,1)}
 backup=BASE/'outputs/before_fig1_reference_20261010';backup.mkdir(exist_ok=True)
 for ext in ['pdf','png','svg']:
  p=OUT/f'figure1_pipeline.{ext}'
  if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
 fig=plt.figure(figsize=(18,13.5));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,100),ylim=(0,75));ax.axis('off')
 text(.7,73.1,'a',28,'bold',ha='left');text(4,73.1,'Data curation and cohort construction (ChEMBL37)',21,'bold',ha='left')
 boxes=[(1,56,16.5,15.2,BLUE,'ChEMBL37\nSQLite archive'),(20,56,21,15.2,GREEN,'Curation & filtering'),(44,56,18.5,15.2,BLUE,'Pair construction'),(65.5,56,17.5,15.2,PEACH,'Filtered repeats'),(86,56,13,15.2,PURPLE,'Evaluation splits')]
 for x,y,w,h,c,title in boxes:card(x,y,w,h,c,title,14)
 for a,b in [(17.7,19.8),(41.2,43.8),(62.7,65.3),(83.2,85.8)]:arrow((a,63.5),(b,63.5))
 # Database cylinder.
 cx,cy=9.2,64.5
 ax.add_patch(Rectangle((cx-2.4,cy-2.3),4.8,3.7,facecolor=BLUE,edgecolor='none',zorder=5))
 for yy in [cy-2.3,cy-1,cy+.3,cy+1.5]:ax.add_patch(Ellipse((cx,yy),4.8,1.25,facecolor=BLUE if yy!=cy+1.5 else LIGHT,edgecolor='white',lw=1,zorder=6))
 text(9.2,60.5,'1,563,556',17,'bold');text(9.2,58.2,'Human single-protein\nactivity candidates',12)
 checks=['Exact IC₅₀ records','Traceable provenance','Valid structures / units','Human single-protein targets']
 for i,s in enumerate(checks):
  yy=66.9-i*1.55;ax.add_patch(Circle((21.9,yy),.58,facecolor=GREEN,edgecolor='none',zorder=6));text(21.9,yy,'✓',13,'bold',color='white');text(23,yy,s,11.6,ha='left',box=(22.9,yy-.7,40.5,yy+.7))
 ax.plot([21,40],[60.55,60.55],color=GRAY,lw=.8);text(30.5,58.8,'690,683 exact IC₅₀ records',12,'bold',box=(20.4,57,40.6,60.4))
 graph(48.8,65.1,1.75,highlight=True);graph(57.2,65.1,1.75,variant=1,highlight=True);arrow((52.65,65.1),(53.35,65.1));text(53.25,61.9,'Same-target assay matching',11.5);text(53.25,59.6,'153,468 molecule–pair rows',12,'bold');text(53.25,57.8,'83,381 assay-pair IDs',11)
 # Funnel.
 for xx,yy in [(71.8,66.1),(73.4,66.4),(74.4,65.8),(76.1,66.3),(72.8,67.1)]:ax.add_patch(Circle((xx,yy),.23,facecolor=GRAY,edgecolor='white',lw=.4,zorder=6))
 ax.add_patch(Polygon([(71,65.2),(77.3,65.2),(74.8,62.4),(74.8,61.5),(73.6,61.1),(73.6,62.4)],facecolor=BLUE,edgecolor=LIGHT,lw=1,zorder=6));text(74.25,59.7,'79,810 molecule–pair rows',11.8,'bold');text(74.25,57.8,'12,392 supported assay pairs',11)
 for i,col in enumerate([BLUE,LIGHT,GREEN,PEACH]):
  for j in range(3):
   for k in range(3):ax.add_patch(Rectangle((87.7+i*2.45+j*.57,64+k*.62),.5,.55,facecolor=col,edgecolor='white',lw=.4,zorder=6))
 text(92.5,61,'Five scaffold splits',13,'bold');text(92.5,58.5,'Scaffolds held out',12)
 text(.7,53.7,'b',28,'bold',ha='left');text(4,53.7,'Modeling framework and tasks',21,'bold',ha='left')
 # Left inputs and central shared response process.
 card(1,40.6,16.5,11.5,BLUE,'Molecular structure\n+ source potency',13.5);graph(5.4,44.5,2.0);text(12.7,45.2,'Morgan bits\nAssay-A pIC₅₀',11.5)
 card(1,29.8,16.5,9.8,BLUE,'Measured references',13.5)
 for cx,v in [(3.8,2),(9.2,3),(14.6,4)]:graph(cx,33.5,1.1,variant=v)
 text(9.2,30.8,'Actual ChEMBL examples',10.8)
 card(20,29.8,43,22.3,BLUE,'SAGP',16)
 card(21.2,39.9,18.5,8.7,GREEN,'Potency-axis mean',13)
 curve(23.3,41.5,14.4,4.5,'mean');text(30.5,40.7,'Source-potency axis',10.5)
 card(21.2,31,18.5,8,GREEN,'Chemical residual',13);text(30.4,35.9,'Molecular Tanimoto kernel',10.5)
 for j in range(7):
  for k in range(4):ax.add_patch(Rectangle((25.6+j*1.3,31.7+k*.73),1.22,.68,facecolor=plt.cm.Blues(.2+.65*np.exp(-abs(j-k)/3)),edgecolor='white',lw=.3,zorder=6))
 graph(23.1,33.2,.9)
 ax.plot([39.8,41.1,41.1],[44.1,44.1,42.1],color=BLUE,lw=1.5,zorder=5);arrow((41.1,42.1),(41.1,41.9),scale=11)
 ax.plot([39.8,41.1,41.1],[34.7,34.7,38.7],color=BLUE,lw=1.5,zorder=5);arrow((41.1,38.7),(41.1,38.9),scale=11)
 ax.add_patch(Circle((41.1,40.4),1.4,facecolor='white',edgecolor=BLUE,lw=1.2,zorder=7));text(41.1,40.4,'+',26);arrow((42.7,40.4),(44,40.4),scale=14)
 card(44.1,38.6,17.7,10,BLUE,'GP mean + covariance',12.8);curve(45.5,40.2,15,5.7,'posterior');text(53,39.3,'Shared posterior (schematic)',9.5)
 document(45.5,32,2.2,3.6);text(54.8,36.5,'Reference summaries',12,'bold');text(54.2,33.9,'Neighborhood similarity\nReference dispersion\nPosterior uncertainty',10.8)
 arrow((17.7,46),(19.8,46));arrow((17.7,34.7),(19.8,34.7))
 # Detached heads and task cards. No protocol category/ESM model inputs.
 card(65.2,33.6,17.4,18.5,BLUE,'Detached decision adapters',12.5)
 card(66.4,43.2,15,5.5,BLUE);graph(68.9,45.4,1.0);text(75,46.3,'SAR risk',13,'bold');text(75,44.6,'Task 1',11)
 card(66.4,35,15,6.5,GREEN);curve(67.1,35.6,4,3.5,'posterior',GREEN);text(75.8,38.8,'Sensitivity',12.5,'bold');text(75.8,37,'Magnitude + risk',10.5)
 arrow((63.1,41.6),(64.1,41.6));ax.plot([64.1,64.1],[38,46],color=BLUE,lw=1.8,zorder=5);arrow((64.1,46),(66.1,46));arrow((64.1,38),(66.1,38));text(73.9,49.4,'stop ∇',10.5,color=RED)
 card(86,43.2,13,7.2,BLUE,'Task 1',15);text(92.5,47,'SAR transfer',13);graph(89.5,44.8,1.0)
 for i,h in enumerate([.5,.8,1.25,1.8]):ax.add_patch(Rectangle((94+i*.8,44),.55,h,facecolor=BLUE,edgecolor=NAVY,lw=.5,zorder=6))
 card(86,34.6,13,7.2,GREEN,'Task 2',15);text(92.5,38.6,'Molecular sensitivity',11.8);graph(89.3,36.2,.82);curve(93,35.2,4.6,2,'posterior',GREEN)
 arrow((82.8,46),(85.8,46));arrow((82.8,38),(85.8,38),GREEN)
 text(41.5,30.4,'Continuous losses → GP',10)
 text(74,31.6,'Risk / magnitude losses → adapters',10)
 # Both labels use observed B only in training/evaluation, never as query input.
 card(1,22.4,16.5,6.3,BLUE,'Training targets',12.3)
 text(9.2,24.2,'Observed B → labels only',10)
 card(20,22.4,43,6.3,BLUE,'Task 1: analogue transfer',12.3)
 text(41.5,24.4,'B-gap = pIC₅₀,B(i) − pIC₅₀,B(j)',12)
 text(41.5,23,'Robust ordering / loss of ordering',10)
 card(65.2,22.4,33.8,6.3,GREEN,'Task 2: molecular sensitivity',12.3)
 text(82.1,24.4,'r = |(B − A) − training-pair median|',11)
 text(82.1,23,'Magnitude + risk at r ≥ 0.5',10)
 text(.7,20.5,'c',28,'bold',ha='left');text(4,20.5,'Representative data patterns',21,'bold',ha='left')
 xs=[5,29,53,79];widths=[17,17,19,19];subheads=['i  Assay potency correlation','ii  Signed shift (B − A)','iii  Analogue agreement','iv  Within-pair support'];axes=[]
 for x,w,title in zip(xs,widths,subheads):text(x,18.6,title,13.8,'bold',ha='left');a=fig.add_axes([x/100,5/75,w/100,12/75]);a.tick_params(labelsize=10,length=3);axes.append(a)
 a=axes[0];a.scatter(g.p_A,g.p_B,s=7,color=BLUE,alpha=.65,edgecolors='none',rasterized=True);lo=min(g.p_A.min(),g.p_B.min())-.1;hi=max(g.p_A.max(),g.p_B.max())+.1;a.plot([lo,hi],[lo,hi],'--',color=GRAY,lw=1);a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Assay A (pIC₅₀)',ylabel='Assay B (pIC₅₀)');a.text(.03,.98,f'n = {len(g)}; ρ = {spearmanr(g.p_A,g.p_B).statistic:.2f}',transform=a.transAxes,va='top',fontsize=11)
 a=axes[1];a.scatter(g.p_A,g.signed_delta,s=7,color=PEACH,alpha=.7,edgecolors='none',rasterized=True);a.axhline(0,ls='--',color=GRAY,lw=.8);a.set(xlabel='Assay A (pIC₅₀)',ylabel='B − A shift (pIC₅₀)');a.text(.03,.98,f'Median shift: {g.signed_delta.median():+.2f}',transform=a.transAxes,va='top',fontsize=11)
 a=axes[2]
 for key,c in [('retained',BLUE),('reversed',RED),('attenuated_or_uncertain',SAND)]:
  h=q[q.outcome.eq(key)];a.scatter(h.anchor_gap_abs,h.oriented_B,s=5,color=c,alpha=.5,edgecolors='none',rasterized=True)
 lim=max(q.anchor_gap_abs.max(),q.oriented_B.max())*1.05;a.plot([0,lim],[0,lim],'--',color=GRAY,lw=.9);a.axhline(0,ls='--',color=GRAY,lw=.8);a.set(xlabel='Assay-A gap (pIC₅₀)',ylabel='A-oriented B gap (pIC₅₀)');a.text(.03,.98,f'n = {len(q):,}; loss = {q.binary_loss.mean():.0%}',transform=a.transAxes,va='top',fontsize=11)
 a=axes[3]
 for source,c,label in [(raw,BLUE,'Complete IC₅₀'),(obs,PEACH,'Filtered repeats')]:
  sizes=source.groupby('pair_id').size();xx=np.arange(1,sizes.max()+1);yy=np.array([(sizes>=i).sum() for i in xx]);a.plot(xx,yy,color=c,lw=1.8,label=label)
 a.set(xscale='log',yscale='log',xlabel='Shared molecules per pair',ylabel='Assay-pair count');a.legend(fontsize=9,loc='upper right',frameon=False);a.text(.04,.07,f'{(raw.groupby("pair_id").size()==1).mean():.0%} singletons',transform=a.transAxes,fontsize=11)
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=s) for s,c in [('Retained',BLUE),('Reversed',RED),('Attenuated / uncertain',SAND)]],loc='lower center',bbox_to_anchor=(.62,.005),ncol=3,frameon=False,fontsize=11)
 # Verify specified card labels remain inside their allotted regions.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();violations=[]
 for t,b in TEXTS:
  bb=t.get_window_extent(renderer);p0=ax.transData.transform((b[0],b[1]));p1=ax.transData.transform((b[2],b[3]));
  if bb.x0<p0[0]-2 or bb.x1>p1[0]+2 or bb.y0<p0[1]-2 or bb.y1>p1[1]+2:violations.append(t.get_text())
 assert not violations,violations
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'figure1_pipeline.{ext}',dpi=300,bbox_inches='tight',pad_inches=.04)
 plt.close(fig);g.to_csv(AUDIT/'reference_pair_molecules.csv',index=False);q.to_csv(AUDIT/'reference_pair_analogues.csv',index=False)
 (AUDIT/'manifest.json').write_text(json.dumps({'reference':'ref.png; layout only','layout':'three rows; five curation modules; one shared GP and two task readouts; four observed-data panels','scientific_corrections':'153468 and79810 are molecule-pair observations, not assay-pair counts; fingerprints/source potency/references are inputs, no assay-category/ESM features; two sensitivity heads; support curve is survival count','molecular_structures':'Single-colour vector circles and lines preserve validated ChEMBL molecular connectivity; elements and bond order are intentionally abstracted; no labels, gradients or highlights in molecular icons','text_bounds_check':'passed','example_pair':chosen,'molecules':len(g),'analogue_comparisons':len(q),'illustrative_molecule_pair':MOL_IDS,'illustrative_pair_assay_pair_id':illustrative_pair_id,'illustrative_pair_support':int(len(display_obs)),'molecule_pair_tanimoto':float(pair.tanimoto),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));print('Saved Figure1; bounded text checks passed')
if __name__=='__main__':main()
