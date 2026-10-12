"""Vector chemical casebook: source-verified structures and frozen predictions."""
from pathlib import Path
import io,json,hashlib,sqlite3,shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cairosvg,re
from sklearn.metrics import roc_auc_score
B=Path(__file__).resolve().parents[1];OUT=B/'outputs/figures';AUDIT=OUT/'chemical_casebook_20261012';AUDIT.mkdir(exist_ok=True)
SRC=OUT/'sensitivity_focus_20261012';sel=pd.read_csv(SRC/'selected_case_predictions.csv');queries=pd.read_csv(SRC/'case_pair_query_predictions_seed17.csv')
BLUE='#649cc9';RED='#e1585d';PEACH='#f29f7b';GRAY='#70777d';INK='#202629';LIGHT='#e8ebed'
W,H=2180,2540;NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
root=ET.Element('{'+NS+'}svg',{'width':'10.9in','height':'12.7in','viewBox':f'0 0 {W} {H}'})
def elem(tag,attrs,parent=root):return ET.SubElement(parent,'{'+NS+'}'+tag,{k:str(v) for k,v in attrs.items()})
elem('rect',dict(x=0,y=0,width=W,height=H,fill='white'))
textboxes=[]
def text(x,y,s,size=24,color=INK,weight='normal',anchor='start'):
 e=elem('text',{'x':x,'y':y,'font-family':'DejaVu Sans','font-size':size,'fill':color,'font-weight':weight,'text-anchor':anchor});e.text=str(s);textboxes.append((x,y,s,size))
def line(x1,y1,x2,y2,c=LIGHT,w=1):elem('line',dict(x1=x1,y1=y1,x2=x2,y2=y2,stroke=c,**{'stroke-width':w}))
EMBED_COUNT=0
def embed(svg,x,y,w,h):
 global EMBED_COUNT
 EMBED_COUNT+=1
 sub=ET.fromstring(svg)
 if 'viewBox' not in sub.attrib:sub.set('viewBox','0 0 480 252')
 ids={e.get('id'):f'panel{EMBED_COUNT}_'+e.get('id') for e in sub.iter() if e.get('id')}
 for e in sub.iter():
  for k,v in list(e.attrib.items()):
   if k=='id':e.set(k,ids[v]);continue
   if v.startswith('#') and v[1:] in ids:e.set(k,'#'+ids[v[1:]])
   else:
    for old,new in ids.items():v=v.replace('url(#'+old+')','url(#'+new+')')
    e.set(k,v)
 vx,vy,vw,vh=map(float,sub.get('viewBox').split());scale=min(w/vw,h/vh);group=elem('g',{'transform':f'translate({x+(w-vw*scale)/2},{y+(h-vh*scale)/2}) scale({scale}) translate({-vx},{-vy})'});
 for child in list(sub):group.append(child)
def rgb(h):return tuple(int(h[i:i+2],16)/255 for i in [1,3,5])
def structure(smiles):
 m=Chem.MolFromSmiles(smiles);assert m is not None
 rdDepictor.SetPreferCoordGen(True);rdDepictor.Compute2DCoords(m)
 conf=m.GetConformer();xy=np.array([[conf.GetAtomPosition(k).x,conf.GetAtomPosition(k).y] for k in range(m.GetNumAtoms())]);xy-=xy.mean(axis=0)
 _,vectors=np.linalg.eigh(np.cov(xy.T));v=vectors[:,-1];angle=np.arctan2(v[1],v[0]);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 xy=xy@rot
 from rdkit.Geometry import Point3D
 for k,pt in enumerate(xy):conf.SetAtomPosition(k,Point3D(float(pt[0]),float(pt[1]),0))
 d=rdMolDraw2D.MolDraw2DSVG(480,252);o=d.drawOptions();o.clearBackground=False;o.padding=.035;o.bondLineWidth=1.7;o.fixedFontSize=26;o.minFontSize=26;o.maxFontSize=26;o.addStereoAnnotation=False
 o.setAtomPalette({1:rgb(INK),6:rgb(INK),7:rgb(BLUE),8:rgb(RED),9:rgb('#8d9b83'),15:rgb(PEACH),16:rgb('#bda04c'),17:rgb('#8d9b83'),35:rgb(PEACH),53:rgb('#8b80ab')})
 d.DrawMolecule(m);d.FinishDrawing();return d.GetDrawingText(),m
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,'xtick.labelsize':9.6,'ytick.labelsize':9.6,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
def panel_svg(fig):
 buf=io.StringIO();fig.savefig(buf,format='svg',transparent=False);plt.close(fig);return buf.getvalue()
conn=sqlite3.connect('file:'+str(B/'data/raw/chembl_37/chembl_37_sqlite/chembl_37.db')+'?mode=ro',uri=True)
raw=[];mapped=[];stats=[]
contexts=[('ROCK2',''),('ER beta',''),('ALDH1A1','')]
for row,(name,context) in enumerate(contexts):
 y=row*500+28;ss=sel[sel.case.eq(name)].set_index('example');g=queries[queries.pair_id.eq(ss.iloc[0].pair_id)].copy();mols=[]
 text(28,y+10,chr(97+row),42,weight='bold');text(85,y+10,'ERβ' if name=='ER beta' else name,31,weight='bold')
 text(2140,y+9,f'{ss.iloc[0].assay_A} / {ss.iloc[0].assay_B}',20,color=GRAY,anchor='end')
 line(28,y+29,2142,y+29)
 for j,key in enumerate(['L','H']):
  r=ss.loc[key];x=35+j*495;ids=[int(r.activity_ids_A),int(r.activity_ids_B)]
  a=pd.read_sql_query('select a.activity_id,m.chembl_id,a.standard_value,a.standard_units,a.standard_relation,cs.canonical_smiles from activities a join molecule_dictionary m on m.molregno=a.molregno join compound_structures cs on cs.molregno=a.molregno where a.activity_id in (?,?)',conn,params=ids).set_index('activity_id')
  assert len(a)==2 and a.chembl_id.nunique()==1
  assert a.standard_units.eq('nM').all() and a.standard_relation.eq('=').all()
  assert np.allclose([a.loc[ids[0]].standard_value,a.loc[ids[1]].standard_value],[r.IC50_A_nM,r.IC50_B_nM],rtol=1e-7)
  # Exact measured ChEMBL structures are shown, with stereochemistry preserved.
  chembl=a.chembl_id.iloc[0];smiles=a.canonical_smiles.iloc[0];svg,m=structure(smiles);mols.append(m)
  col=BLUE if key=='L' else RED
  elem('circle',dict(cx=x+15,cy=y+74,r=13,fill=col));text(x+15,y+82,'1' if key=='L' else '2',17,'white','bold','middle');text(x+40,y+82,chembl,22,color=GRAY)
  embed(svg,x,y+113,470,252)
  def dose(v):return f'{v:,.0f}' if v>=100 else f'{v:.3g}'
  text(x+12,y+405,f'IC50: {dose(r.IC50_A_nM)} → {dose(r.IC50_B_nM)} nM',27)
  text(x+12,y+450,f'r: {r.target:.3f} → {r.regression:.3f}',27,weight='bold')
  rr=r.to_dict();rr.update(example=key,molecule_chembl_id=chembl,canonical_isomeric_smiles=smiles);mapped.append(rr);raw.extend(a.reset_index().to_dict('records'))

 # Middle: measured source and destination potencies, never model inputs from B.
 fig,ax=plt.subplots(figsize=(2.15,2.03));fig.subplots_adjust(left=.23,right=.96,bottom=.25,top=.88)
 for key,c in [('L',BLUE),('H',RED)]:
  r=ss.loc[key];ax.plot([0,1],[r.p_A,r.p_B],color=c,lw=1.6,marker='o',ms=5,clip_on=False)
  for k,p in enumerate([r.p_A,r.p_B]):ax.annotate(f'{p:.2f}',(k,p),xytext=(0,7 if p==ss[['p_A','p_B']].iloc[:,k].max() else -14),textcoords='offset points',ha='center',color=c,fontsize=8.7)
 vals=ss[['p_A','p_B']].values.flatten();ax.set_ylim(vals.min()-.75,vals.max()+.55);ax.set_xlim(-.20,1.20);ax.set_xticks([0,1],['A','B']);ax.set_xlabel('');ax.set_ylabel('pIC$_{50}$');ax.tick_params(length=3)
 embed(panel_svg(fig),1040,y+107,410,365);text(1060,y+75,'pIC50',25,weight='bold')
 # Right: full selected-pair held-out cohort. No errors omitted.
 fig,ax=plt.subplots(figsize=(3.00,2.30));fig.subplots_adjust(left=.20,right=.97,bottom=.24,top=.95)
 for lab,c in [(0,BLUE),(1,RED)]:
  gg=g[g.binary_loss.eq(lab)];ax.scatter(gg.target,gg.regression,s=25,color=c,alpha=.82,edgecolor='white',linewidth=.5,zorder=3)
 lim=max(g.target.max(),g.regression.max(),.8)*1.10;ax.plot([0,lim],[0,lim],ls='--',lw=.8,color='#9da5aa',zorder=0);ax.axvline(.5,ls=':',lw=.75,color='#aab2b7',zorder=0)
 for key,c in [('L',BLUE),('H',RED)]:
  r=ss.loc[key];col=BLUE if key=='L' else RED;ax.scatter([r.target],[r.regression],s=76,facecolors='none',edgecolors=col,lw=1.4,zorder=4);ax.annotate('1' if key=='L' else '2',(r.target,r.regression),xytext=(7,7),textcoords='offset points',color=col,weight='bold',fontsize=9,zorder=5)
 ax.set(xlim=(-.035,lim),ylim=(-.035,lim),xlabel='Observed r (pIC$_{50}$)',ylabel='Predicted r (pIC$_{50}$)');ax.tick_params(length=3)
 embed(panel_svg(fig),1490,y+106,662,365)
 mae=abs(g.target-g.regression).mean();auc=roc_auc_score(g.binary_loss,g.prediction);text(1540,y+75,f'n {len(g)}  ·  MAE {mae:.3f}  ·  AUROC {auc:.3f}',22,weight='bold')
 stats.append(dict(case=name,n=len(g),mae=float(mae),auroc=float(auc)))
# Task 1 rows use the same vector layout and exact activity-linked structures.
D=B/'outputs/datasets/sar_transfer_20261007'
pred1=pd.read_csv(D/'contextual_decision_gp_20261009/task1_context_neural_selected_test_seed_17.csv')
comparisons=pd.read_parquet(D/'primary_comparisons.parquet')
q1=pred1.merge(comparisons,on=['comparison_id','pair_id','target_chembl_id'],validate='one_to_one')
task1rows=[];task1raw=[];task1cohorts=[]
for row,(kind,cid) in enumerate([('Retained SAR','sar_ce494b6496d0f44f04353b13'),('Reversed SAR','sar_254f41fa9431b8ac3a8150e9')],start=3):
 y=row*500+28;r=q1[q1.comparison_id.eq(cid)].iloc[0];g=q1[q1.pair_id.eq(r.pair_id)].copy()
 assert (r.prediction>=.5)==bool(r.binary_loss)
 assert np.isclose(r.target,np.sign(r.gap_A)*r.gap_B,atol=1e-6)
 text(28,y+10,chr(97+row),42,weight='bold');text(85,y+10,f'Task 1 · {kind}',31,weight='bold')
 text(2140,y+9,f'{r.assay_A} / {r.assay_B}',20,color=GRAY,anchor='end');line(28,y+29,2142,y+29)
 for j in [1,2]:
  x=35+(j-1)*495;aid=int(r[f'activity_ids_A_{j}']);bid=int(r[f'activity_ids_B_{j}'])
  a=pd.read_sql_query('select a.activity_id,m.chembl_id,a.standard_value,a.standard_units,a.standard_relation,cs.canonical_smiles from activities a join molecule_dictionary m on m.molregno=a.molregno join compound_structures cs on cs.molregno=a.molregno where a.activity_id in (?,?)',conn,params=[aid,bid]).set_index('activity_id')
  assert len(a)==2 and a.chembl_id.nunique()==1 and a.standard_units.eq('nM').all() and a.standard_relation.eq('=').all()
  doses=[10**(9-r[f'p_A_{j}']),10**(9-r[f'p_B_{j}'])];assert np.allclose(a.loc[[aid,bid]].standard_value,doses,rtol=1e-7)
  chembl=a.chembl_id.iloc[0];smiles=a.canonical_smiles.iloc[0];sv,m=structure(smiles);col=BLUE if j==1 else RED
  elem('circle',dict(cx=x+15,cy=y+74,r=13,fill=col));text(x+15,y+82,str(j),17,'white','bold','middle');text(x+40,y+82,chembl,22,color=GRAY)
  embed(sv,x,y+113,470,252)
  text(x+12,y+405,f'IC50: {dose(doses[0])} → {dose(doses[1])} nM',27)
  text(x+12,y+450,f'Tanimoto {r.tanimoto:.2f}' if j==1 else f'P(SAR loss) {r.prediction:.2f}',25,weight='bold')
  task1rows.append(dict(case=kind,comparison_id=cid,pair_id=r.pair_id,target_chembl_id=r.target_chembl_id,molecule_chembl_id=chembl,canonical_smiles=smiles,activity_id_A=aid,activity_id_B=bid,IC50_A_nM=doses[0],IC50_B_nM=doses[1],observed_B_gap=r.target,predicted_B_gap=r.regression,loss_probability=r.prediction))
  task1raw.extend(a.reset_index().to_dict('records'))
 # Measured potencies retain the physical ranking change, using the same two compound colours.
 fig,ax=plt.subplots(figsize=(2.15,2.03));fig.subplots_adjust(left=.23,right=.96,bottom=.25,top=.88)
 vv=np.array([[r.p_A_1,r.p_B_1],[r.p_A_2,r.p_B_2]])
 for j,c in [(1,BLUE),(2,RED)]:
  vals=[r[f'p_A_{j}'],r[f'p_B_{j}']];ax.plot([0,1],vals,color=c,lw=1.6,marker='o',ms=5,clip_on=False)
  for k,pot in enumerate(vals):ax.annotate(f'{pot:.2f}',(k,pot),xytext=(0,7 if pot==vv[:,k].max() else -14),textcoords='offset points',ha='center',color=c,fontsize=8.7)
 ax.set_ylim(vv.min()-.75,vv.max()+.55);ax.set_xlim(-.20,1.20);ax.set_xticks([0,1],['A','B']);ax.set_ylabel('pIC$_{50}$');ax.tick_params(length=3)
 embed(panel_svg(fig),1040,y+107,410,365);text(1060,y+75,'pIC50',25,weight='bold')
 fig,ax=plt.subplots(figsize=(3.00,2.30));fig.subplots_adjust(left=.20,right=.97,bottom=.24,top=.95)
 for lab,c in [(0,BLUE),(1,RED)]:
  gg=g[g.binary_loss.eq(lab)];ax.scatter(gg.target,gg.regression,s=25,color=c,alpha=.82,edgecolor='white',linewidth=.5,zorder=3)
 lo=min(g.target.min(),g.regression.min())-.10;hi=max(g.target.max(),g.regression.max())+.10
 ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color='#9da5aa',zorder=0)
 col=RED if r.binary_loss else BLUE;ax.scatter([r.target],[r.regression],s=85,facecolors='none',edgecolors=col,lw=1.4,zorder=4)
 ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed B gap (pIC$_{50}$)',ylabel='Predicted B gap (pIC$_{50}$)');ax.tick_params(length=3)
 embed(panel_svg(fig),1490,y+106,662,365)
 text(1540,y+75,f'n {len(g)} · gap {r.target:+.2f} → {r.regression:+.2f}',22,weight='bold')
 task1cohorts.append(g)
line(28,2518,2142,2518);text(28,2536,'Task 1: A-oriented B gap  ·  Task 2: residual magnitude r',19,color=GRAY);text(2142,2536,'Scaffold-held-out split 17',19,color=GRAY,anchor='end')
T1=OUT/'task1_casebook_20261012';T1.mkdir(exist_ok=True)
pd.DataFrame(task1rows).to_csv(T1/'task1_case_measurement_lineage.csv',index=False)
pd.DataFrame(task1raw).to_csv(T1/'raw_task1_activity_structures.csv',index=False)
pd.concat(task1cohorts).to_csv(T1/'figure4_all_featured_test_queries_seed17.csv',index=False)
(T1/'manifest.json').write_text(json.dumps({'seed':17,'cases':[x['comparison_id'] for x in task1rows[::2]],'structures':'Exact activity-linked ChEMBL37 structures; custom original atom palette','selection':'Retrospectively selected accurate retained and reversed SAR examples; complete selected-pair test cohorts shown'},indent=2))
svg=ET.tostring(root,encoding='unicode');(OUT/'figure4_casebook.svg').write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/'figure4_casebook.pdf'));cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/'figure4_casebook.png'),output_width=3270,output_height=3810)
pd.DataFrame(mapped).to_csv(AUDIT/'selected_case_structures_predictions.csv',index=False);pd.DataFrame(raw).to_csv(AUDIT/'raw_activity_structure_lineage.csv',index=False);queries.to_csv(AUDIT/'complete_selected_pair_test_predictions.csv',index=False)
manifest=dict(cases=stats,selection='Same six retrospectively selected molecules as previous casebook; no further performance filtering.',structures='Exact ChEMBL37 activity-linked canonical structures; stereochemistry and bond orders retained. No experimental conformation or binding pose inferred.',fingerprint='2048-bit radius-2 chirality-aware Morgan; similarity is descriptive, not a single-edit/MMP assertion.',sources={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'selected_case_predictions.csv',SRC/'case_pair_query_predictions_seed17.csv']},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(AUDIT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(stats,indent=2))
