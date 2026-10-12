"""Single neural reference-attention models for SAR and molecular sensitivity.

All retrieved references are training rows. During training, all references
sharing either query scaffold are masked, not just the same comparison.
Held-out partitions use existing/global scaffold assignments. Query B values
are supervised targets only. No averaging of fitted-model predictions.
"""
from __future__ import annotations
import argparse,copy,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from scipy.stats import spearmanr
from xgboost import XGBClassifier,XGBRegressor
import joblib
from prepare_sar_transfer_20261007 import OUT,SEEDS,partition,digest
from train_sar_transfer_baseline_20261007 import read,make_molecule_features,assay_features
from train_anchored_sar_transfer_20261007 import encode,metrics
from train_contrast_boost_sar_20261007 import pair_history,a_uncertainty

RESULTS=OUT/'models_deep_three_tasks_20261007';RESULTS.mkdir(exist_ok=True)
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
K=12

class ReferenceNet(nn.Module):
    def __init__(self,dim,edim,geometry,attention=True):
        super().__init__();self.attention=attention;self.geometry=geometry
        self.query=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.25),
                                 nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU())
        if attention:
            self.score=nn.Sequential(nn.Linear(geometry,32),nn.SiLU(),nn.Linear(32,1))
            self.value=nn.Sequential(nn.Linear(edim,32),nn.SiLU(),nn.Linear(32,32))
        self.head=nn.Sequential(nn.Linear(96 if attention else 64,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,2))
    def forward(self,x,e,mask):
        q=self.query(x)
        if self.attention:
            score=self.score(e[...,:self.geometry]).squeeze(-1)
            score=score.masked_fill(~mask,-1e4)
            weights=score.softmax(-1)*mask
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            ref=(weights[...,None]*self.value(e)).sum(1)
            q=torch.cat([q,ref],-1)
        return self.head(q)


class AnchoredReferenceNet(ReferenceNet):
    def __init__(self,dim,edim,geometry,attention=True,start=0):
        super().__init__(dim-start,edim,geometry,attention)
        self.anchored=True;self.start=start
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,e,mask,prior=None,anchor=None):
        out=super().forward(x[:,self.start:],e,mask)
        # A train-estimated context prior and measured A gap are explicit
        # offsets inside ONE trained model, not outputs from other learners.
        logit=torch.logit(prior.clamp(.005,.995))
        return out+torch.stack([logit,anchor],-1)


class NeuralEmpiricalBayesNet(AnchoredReferenceNet):
    """One jointly learned local Beta model, not fitted-model fusion.

    Reference labels are permitted observations. A neural chemical kernel
    learns their effective evidence weights; the context network learns a
    Beta prior and its concentration. Continuous targets use the same
    evidence weights and a learned prior location.
    """
    def __init__(self,dim,edim,geometry,task):
        super().__init__(dim,edim,geometry,attention=False,start=0)
        self.task=task
        self.evidence=nn.Sequential(nn.Linear(geometry,24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.evidence[-1].weight);nn.init.zeros_(self.evidence[-1].bias)
        self.concentration=nn.Linear(64,1)
        nn.init.zeros_(self.concentration.weight);nn.init.constant_(self.concentration.bias,1.5)
    def forward(self,x,e,mask,prior=None,anchor=None):
        q=self.query(x);delta=self.head(q)
        prior_risk=(torch.logit(prior.clamp(.005,.995))+delta[:,0]).sigmoid()
        kappa=F.softplus(self.concentration(q).squeeze(-1))+1.
        similarity=(e[...,0].clamp(0,1)*e[...,1].clamp(0,1)).sqrt() if self.task==1 else e[...,0].clamp(0,1)
        weight=10*similarity.pow(4)*torch.exp(self.evidence(e[...,:self.geometry]).squeeze(-1).clamp(-4,4))*mask
        labels=e[...,4] if self.task==1 else e[...,2]
        denominator=kappa+weight.sum(-1)
        risk=(kappa*prior_risk+(weight*labels).sum(-1))/denominator
        prior_mean=anchor+delta[:,1]
        if self.task==1:
            reference_mean=3*e[...,5]+2*e[...,2]
        else:
            prior_mean=F.softplus(delta[:,1]+torch.log(torch.expm1(anchor.clamp_min(.01))))
            reference_mean=3*e[...,3]
        mean=(kappa*prior_mean+(weight*reference_mean).sum(-1))/denominator
        return torch.stack([torch.logit(risk.clamp(.00001,.99999)),mean],-1)


class CalibratedDistributionNet(AnchoredReferenceNet):
    """Single conditional logistic distribution with a calibrated mean.

    For SAR, mean is the predicted median B gap and a supervised nonnegative
    replicate-width term translates it to the robust-margin event. For
    molecular sensitivity, the two tails of a signed residual are evaluated.
    """
    def __init__(self,dim,edim,geometry,task):
        super().__init__(dim,edim,geometry,attention=True,start=0)
        self.task=task;self.distributional=True
        self.head[-1]=nn.Linear(64,3)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[2]=-4.
    def forward(self,x,e,mask,prior=None,anchor=None,noise=None,signed_anchor=None):
        q=self.query(x)
        score=self.score(e[...,:self.geometry]).squeeze(-1).masked_fill(~mask,-1e4)
        weight=score.softmax(-1)*mask;weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        value=(weight[...,None]*self.value(e)).sum(1)
        delta=self.head(torch.cat([q,value],-1))
        normal_sd=noise.clamp_min(.03)*torch.exp(2*torch.tanh(delta[:,1]/2))
        logistic_scale=normal_sd*math.sqrt(3)/math.pi
        if self.task==1:
            mean=anchor+delta[:,0];self.last_width=F.softplus(delta[:,2])
            logits=(.3-mean+self.last_width)/logistic_scale
            return torch.stack([logits,mean],-1)
        mu=signed_anchor+delta[:,0]
        prob=((mu-.5)/logistic_scale).sigmoid()+((-mu-.5)/logistic_scale).sigmoid()
        z=mu/normal_sd
        absolute_mean=normal_sd*math.sqrt(2/math.pi)*torch.exp(-.5*z.square())+mu*torch.erf(z/math.sqrt(2))
        return torch.stack([torch.logit(prob.clamp(1e-6,1-1e-6)),absolute_mean],-1)


def predict(net,arrays):
    net.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(arrays['y']),1024):
            args=[arrays['x'][i:i+1024],arrays['edge'][i:i+1024],arrays['mask'][i:i+1024]]
            if getattr(net,'anchored',False):args.extend([arrays['prior'][i:i+1024],arrays['anchor'][i:i+1024]])
            if getattr(net,'distributional',False):args.extend([arrays['noise'][i:i+1024],arrays['signed_anchor'][i:i+1024]])
            o=net(*args)
            out.append(torch.stack([o[:,0].sigmoid(),o[:,1]],-1).cpu().numpy())
    return np.concatenate(out)


def sensitivity_metrics(frame,pred,reg):
    y=frame.binary_loss.to_numpy();valid=[];enrichment=[]
    for _,ix in frame.groupby('pair_id').indices.items():
        yy=y[ix]
        if len(ix)>=10 and yy.sum()>=3 and (1-yy).sum()>=3:
            valid.append((roc_auc_score(yy,pred[ix]),average_precision_score(yy,pred[ix])))
            top=np.argsort(-pred[ix])[:max(1,int(np.ceil(.1*len(ix))))]
            enrichment.append(yy[top].mean()/yy.mean())
    err=abs(frame.residual_abs.to_numpy()-reg)
    macro=pd.DataFrame({'pair_id':frame.pair_id.to_numpy(),'err':err}).groupby('pair_id').err.mean().mean()
    return {'n':len(y),'pairs':frame.pair_id.nunique(),'prevalence':float(y.mean()),
            'AP':float(average_precision_score(y,pred)),
            'AUROC':float(roc_auc_score(y,pred)),
            'Brier':float(brier_score_loss(y,pred)),
            'pair_macro_AUROC':float(np.mean([v[0] for v in valid])) if valid else np.nan,
            'pair_macro_AP':float(np.mean([v[1] for v in valid])) if valid else np.nan,
            'pair_macro_n':len(valid),'within_pair_top10_enrichment':float(np.mean(enrichment)) if enrichment else np.nan,
            'MAE':float(err.mean()),'macro_MAE':float(macro),
            'spearman':float(spearmanr(frame.residual_abs,reg).statistic)}


def eval_metrics(task,frame,pred,reg):
    if task==2:return sensitivity_metrics(frame,pred,np.maximum(reg,0))
    m=metrics(frame,pred);err=abs(frame.target_gap.to_numpy()-reg)
    m['MAE']=float(err.mean());m['macro_MAE']=float(pd.DataFrame({'pair_id':frame.pair_id.to_numpy(),'err':err}).groupby('pair_id').err.mean().mean())
    return m


def train(arrays,frames,task,seed,attention,epochs,anchored=False,projected=False,empirical_bayes=False,distributional=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tensors={p:{k:torch.as_tensor(v,device=DEVICE) for k,v in a.items()} for p,a in arrays.items()}
    tr=tensors['train']
    if distributional:
        net=CalibratedDistributionNet(tr['x'].shape[1],tr['edge'].shape[2],4 if task==1 else 2,task).to(DEVICE)
    elif empirical_bayes:
        net=NeuralEmpiricalBayesNet(tr['x'].shape[1],tr['edge'].shape[2],4 if task==1 else 2,task).to(DEVICE)
    elif anchored:
        start=0 if projected else (1024 if task==1 else 512)
        net=AnchoredReferenceNet(tr['x'].shape[1],tr['edge'].shape[2],4 if task==1 else 2,attention,start).to(DEVICE)
    else:net=ReferenceNet(tr['x'].shape[1],tr['edge'].shape[2],4 if task==1 else 2,attention).to(DEVICE)
    optim=torch.optim.AdamW(net.parameters(),lr=.0003 if anchored else .0007,weight_decay=.003)
    best=-np.inf;state=None;best_epoch=0;hist=[]
    # Loss weights prevent one large assay context from dominating molecular
    # ranking while retaining a row-wise objective.
    freq=frames['train'].pair_id.value_counts()
    weight=1/np.sqrt(frames['train'].pair_id.map(freq).to_numpy(dtype=np.float32))
    weight=weight/weight.mean();wt=torch.tensor(weight,device=DEVICE)
    if anchored and task==1:wt=torch.ones_like(wt)
    pair_code=torch.tensor(pd.factorize(frames['train'].pair_id)[0],device=DEVICE)
    for epoch in range(epochs):
        net.train();perm=torch.tensor(rng.permutation(len(tr['y'])),device=DEVICE);losses=[]
        for ix in perm.split(512):
            args=[tr['x'][ix],tr['edge'][ix],tr['mask'][ix]]
            if anchored:args.extend([tr['prior'][ix],tr['anchor'][ix]])
            if distributional:args.extend([tr['noise'][ix],tr['signed_anchor'][ix]])
            out=net(*args)
            loss=(F.binary_cross_entropy_with_logits(out[:,0],tr['y'][ix],reduction='none')+
                  .2*F.smooth_l1_loss(out[:,1],tr['target'][ix],reduction='none'))
            loss=(loss*wt[ix]).mean()
            if distributional and task==1:loss=loss+.1*F.smooth_l1_loss(net.last_width,tr['width_target'][ix])
            if projected:
                same=pair_code[ix,None]==pair_code[ix][None,:]
                posneg=same&(tr['y'][ix,None]>tr['y'][ix][None,:])
                if posneg.any():
                    delta=out[:,0,None]-out[:,0][None,:]
                    loss=loss+.1*F.softplus(-delta[posneg]).mean()
            optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
            losses.append(float(loss.detach()))
        pv=predict(net,tensors['validation']);m=eval_metrics(task,frames['validation'],pv[:,0],pv[:,1])
        score=m['AP'] if task==1 else m['pair_macro_AUROC']
        if not np.isfinite(score):score=m['AP']
        hist.append({'epoch':epoch+1,'loss':np.mean(losses),'val_selection':score,'val_AP':m['AP']})
        if score>best+1e-5:best=score;state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=12:break
    net.load_state_dict(state);pred={p:predict(net,a) for p,a in tensors.items() if p!='train'}
    return net,pred,hist,best_epoch


def sim(a,b):
    dot=a@b.T
    return dot/(a.sum(1)[:,None]+b.sum(1)[None,:]-dot).clamp_min(1e-6)


def reference_arrays(task,frames,mol):
    tr=frames['train'];dim=7 if task==1 else 6
    arrays={p:{'edge':np.zeros((len(f),K,dim),np.float32),'mask':np.zeros((len(f),K),bool),
               'local_prediction':np.full(len(f),float(tr.binary_loss.mean()),np.float32),
               'local_reg':np.full(len(f),float(tr.target.mean()),np.float32),
               'max_similarity':np.zeros(len(f),np.float32)} for p,f in frames.items()}
    train_groups=tr.groupby('pair_id').indices
    for part,frame in frames.items():
        for pair,qi in frame.groupby('pair_id').indices.items():
            ti=train_groups.get(pair)
            if ti is None:continue
            support=tr.iloc[ti];query=frame.iloc[qi]
            if task==1:
                ts=torch.tensor(np.stack([mol[i][0] for i in support.strong]),device=DEVICE)
                tw=torch.tensor(np.stack([mol[i][0] for i in support.weak]),device=DEVICE)
            else:ts=torch.tensor(np.stack([mol[i][0] for i in support.chemical_identity]),device=DEVICE)
            for start in range(0,len(qi),256):
                ix=qi[start:start+256];q=frame.iloc[ix]
                if task==1:
                    qs=torch.tensor(np.stack([mol[i][0] for i in q.strong]),device=DEVICE)
                    qw=torch.tensor(np.stack([mol[i][0] for i in q.weak]),device=DEVICE)
                    s1=sim(qs,ts);s2=sim(qw,tw);score=(s1*s2).sqrt()
                else:
                    qs=torch.tensor(np.stack([mol[i][0] for i in q.chemical_identity]),device=DEVICE)
                    score=sim(qs,ts)
                if part=='train':
                    if task==1:
                        allowed=(q.scaffold_1.to_numpy()[:,None]!=support.scaffold_1.to_numpy()[None,:])&\
                                (q.scaffold_1.to_numpy()[:,None]!=support.scaffold_2.to_numpy()[None,:])&\
                                (q.scaffold_2.to_numpy()[:,None]!=support.scaffold_1.to_numpy()[None,:])&\
                                (q.scaffold_2.to_numpy()[:,None]!=support.scaffold_2.to_numpy()[None,:])
                    else:allowed=q.scaffold.to_numpy()[:,None]!=support.scaffold.to_numpy()[None,:]
                    score=score.masked_fill(~torch.tensor(allowed,device=DEVICE),-1)
                vals,ids=score.topk(min(K,len(ti)),dim=1)
                vals=vals.cpu().numpy();ids=ids.cpu().numpy();nk=ids.shape[1];ok=vals>=0
                a=arrays[part];a['mask'][ix,:nk]=ok
                sy=support.binary_loss.to_numpy()[ids];sr=support.target.to_numpy()[ids]
                if task==1:
                    s1=s1.cpu().numpy();s2=s2.cpu().numpy();rows=np.arange(len(ix))[:,None]
                    edge=np.stack([s1[rows,ids],s2[rows,ids],
                         (q.anchor_gap_abs.to_numpy()[:,None]-support.anchor_gap_abs.to_numpy()[ids])/2,
                         (q.anchor_p_mean.to_numpy()[:,None]-support.anchor_p_mean.to_numpy()[ids])/2,
                         sy,sr/3,vals],-1)
                else:
                    edge=np.stack([vals,(q.p_A.to_numpy()[:,None]-support.p_A.to_numpy()[ids])/2,
                         sy,sr/3,support.raw_label_ge_1.to_numpy()[ids],
                         (support.max_A-support.min_A).to_numpy()[ids]],-1)
                a['edge'][ix,:nk]=np.where(ok[...,None],edge,0)
                w=np.maximum(vals,0)**4*ok;den=w.sum(1)
                a['local_prediction'][ix]=np.clip(np.where(den>0,(w*sy).sum(1)/np.maximum(den,1e-8),float(tr.binary_loss.mean())),0,1)
                a['local_reg'][ix]=np.where(den>0,(w*sr).sum(1)/np.maximum(den,1e-8),float(tr.target.mean()))
                a['max_similarity'][ix]=np.maximum(vals[:,0],0)
            del ts
            if task==1:del tw
    return arrays


def assay_block(frame,av,median):
    def block(ids):
        rows=[av[i] for i in ids];nums=np.stack([r[2] for r in rows]);observed=np.isfinite(nums).astype(np.float32)
        return np.concatenate([np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows]),
                               np.where(np.isfinite(nums),nums,median),observed],1)
    a=block(frame.assay_A);b=block(frame.assay_B)
    return np.concatenate([a,b,b-a],1).astype(np.float32)


def prepare_task1(seed,data,mol,include_references=True):
    split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
    frames={p:data.merge(split[split.partition.eq(p)][['comparison_id']],on='comparison_id',validate='one_to_one') for p in ['train','validation','test']}
    for f in frames.values():
        pos=f.gap_A>0;f['strong']=np.where(pos,f.identity_1,f.identity_2);f['weak']=np.where(pos,f.identity_2,f.identity_1)
        f['target_gap']=f.gap_B*np.where(pos,1,-1);f['target']=f.target_gap
    av,med,vect,svd,formats=assay_features(frames['train'],pd.concat(frames.values()))
    targ={i:np.zeros(1,np.float32) for i in data.target_chembl_id.unique()}
    arrays=reference_arrays(1,frames,mol) if include_references else {p:{} for p in frames}
    for p,f in frames.items():
        blocks=encode(f,mol,av,med,targ)
        h=pair_history(frames['train'],f,loo=p=='train')
        # Pair prior is metadata learned from training outcomes, with the
        # query label removed in training. All retrieved references have the
        # stricter scaffold-exclusion rule above.
        arrays[p]['x']=np.concatenate([blocks['molecule'],blocks['assay'],a_uncertainty(f),h],1)
        arrays[p]['y']=f.binary_loss.to_numpy(np.float32);arrays[p]['target']=f.target.to_numpy(np.float32)
        arrays[p]['prior']=h[:,0];arrays[p]['anchor']=f.anchor_gap_abs.to_numpy(np.float32)
    scaler=StandardScaler().fit(arrays['train']['x'])
    for a in arrays.values():a['x']=np.clip(scaler.transform(a['x']),-8,8).astype(np.float32)
    pre={'scaler':scaler,'tfidf':vect,'svd':svd,'formats':formats,'numeric_median':med}
    return frames,arrays,pre


def prepare_task2(seed,obs,mol,include_references=True):
    assignment={s:partition(s,seed) for s in obs.scaffold.unique()}
    tr=obs[obs.scaffold.map(assignment).eq('train')]
    known=set(tr.groupby('pair_id').size().loc[lambda x:x>=30].index)
    frames={p:obs[obs.scaffold.map(assignment).eq(p)&obs.pair_id.isin(known)].reset_index(drop=True).copy() for p in ['train','validation','test']}
    center=frames['train'].groupby('pair_id').signed_delta.median()
    for f in frames.values():
        f['pair_offset']=f.pair_id.map(center);f['residual_abs']=abs(f.signed_delta-f.pair_offset)
        f['binary_loss']=(f.residual_abs>=.5).astype(int);f['target']=f.residual_abs
    av,med,vect,svd,formats=assay_features(frames['train'],pd.concat(frames.values()))
    arrays=reference_arrays(2,frames,mol) if include_references else {p:{} for p in frames}
    stats=frames['train'].groupby('pair_id').binary_loss.agg(['sum','count'])
    for p,f in frames.items():
        fp=np.stack([mol[i][0] for i in f.chemical_identity]);desc=np.stack([mol[i][1] for i in f.chemical_identity])
        # Exclude the query outcome from the training classification prior.
        successes=f.pair_id.map(stats['sum']).to_numpy();n=f.pair_id.map(stats['count']).to_numpy()
        if p=='train':successes=successes-f.binary_loss.to_numpy();n=n-1
        prior=(successes+20*float(frames['train'].binary_loss.mean()))/(n+20)
        extra=np.column_stack([f.p_A,(f.max_A-f.min_A),prior,np.log1p(n)])
        arrays[p]['x']=np.concatenate([fp,desc,assay_block(f,av,med),extra],1).astype(np.float32)
        arrays[p]['y']=f.binary_loss.to_numpy(np.float32);arrays[p]['target']=f.target.to_numpy(np.float32)
        arrays[p]['pair_prior']=prior
        arrays[p]['prior']=prior.astype(np.float32)
        arrays[p]['anchor']=np.full(len(f),float(frames['train'].residual_abs.median()),np.float32)
    scaler=StandardScaler().fit(arrays['train']['x'])
    for a in arrays.values():a['x']=np.clip(scaler.transform(a['x']),-8,8).astype(np.float32)
    pre={'scaler':scaler,'tfidf':vect,'svd':svd,'formats':formats,'numeric_median':med,'pair_offset':center}
    return frames,arrays,pre


def audit(frames,task):
    cols=['scaffold_1','scaffold_2'] if task==1 else ['scaffold']
    for p in ['validation','test']:
        left=set().union(*(set(frames['train'][c]) for c in cols));right=set().union(*(set(frames[p][c]) for c in cols))
        assert left.isdisjoint(right)
    if task==2:
        assert frames['train'].groupby('pair_id').size().min()>=30
        for field in ['measurement_id_A','measurement_id_B','provenance_equivalence_groups_A','provenance_equivalence_groups_B']:
            left={v for x in frames['train'][field].astype(str) for v in x.split(';')}
            for p in ['validation','test']:
                right={v for x in frames[p][field].astype(str) for v in x.split(';')}
                assert left.isdisjoint(right),(field,p)


def save_prediction(task,seed,name,part,frame,pred,reg,maxsim):
    idcol='comparison_id' if task==1 else 'sample_id'
    out=frame[[idcol,'pair_id','target_chembl_id','binary_loss']].copy()
    out['prediction']=pred;out['target']=frame.target;out['regression']=np.maximum(reg,0) if task==2 else reg;out['max_reference_similarity']=maxsim
    out.to_csv(RESULTS/f'task{task}_{name}_{part}_seed_{seed}.csv',index=False)


def main(seeds,tasks,epochs,kinds):
    torch.set_num_threads(4)
    data=read('anchor_transfer_all.parquet');obs=read('molecule_assay_observations.parquet')
    mol=make_molecule_features(set(obs.chemical_identity)|set(data.identity_1)|set(data.identity_2))
    path=RESULTS/'metrics.csv';rows=[];counts=[]
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.task.isin(tasks)&old.model.isin(kinds))].to_dict('records')
    oldcounts=RESULTS/'split_counts.csv'
    if oldcounts.exists():
        c=pd.read_csv(oldcounts);counts=c[~(c.seed.isin(seeds)&c.task.isin(tasks))].to_dict('records')
    for task in tasks:
        for seed in seeds:
            start=time.monotonic()
            frames,arrays,pre=(prepare_task1(seed,data,mol) if task==1 else prepare_task2(seed,obs,mol))
            audit(frames,task)
            for p,f in frames.items():counts.append({'task':task,'seed':seed,'partition':p,'rows':len(f),'pairs':f.pair_id.nunique(),'positives':int(f.binary_loss.sum())})
            joblib.dump(pre,RESULTS/f'task{task}_preprocess_seed_{seed}.joblib',compress=3)
            for p in ['validation','test']:
                f=frames[p];a=arrays[p]
                baseline=[('local_knn',a['local_prediction'],a['local_reg'])]
                if task==2:
                    med=frames['train'].groupby('pair_id').residual_abs.median()
                    baseline.append(('pair_prior',a['pair_prior'],f.pair_id.map(med).to_numpy()))
                else:baseline.append(('retain_gap_A',np.full(len(f),float(frames['train'].binary_loss.mean())),f.anchor_gap_abs.to_numpy()))
                for name,pred,reg in baseline:
                    if any(r['task']==task and r['seed']==seed and r['model']==name and r['partition']==p for r in rows):continue
                    rows.append({'task':task,'seed':seed,'model':name,'partition':p,**eval_metrics(task,f,pred,reg)})
                    save_prediction(task,seed,name,p,f,pred,reg,a['max_similarity'])
            if task==2 and not any(r['task']==2 and r['seed']==seed and r['model']=='xgboost' for r in rows):
                tr=arrays['train'];va=arrays['validation']
                clf=XGBClassifier(n_estimators=500,max_depth=4,min_child_weight=15,learning_rate=.04,reg_lambda=10,subsample=.85,colsample_bytree=.85,n_jobs=4,tree_method='hist',early_stopping_rounds=30,random_state=seed)
                clf.fit(tr['x'],tr['y'],eval_set=[(va['x'],va['y'])],verbose=False)
                reg=XGBRegressor(n_estimators=500,max_depth=4,min_child_weight=15,learning_rate=.04,reg_lambda=10,n_jobs=4,tree_method='hist',early_stopping_rounds=30,random_state=seed)
                reg.fit(tr['x'],tr['target'],eval_set=[(va['x'],va['target'])],verbose=False)
                for p in ['validation','test']:
                    a=arrays[p];pp=clf.predict_proba(a['x'])[:,1];rr=reg.predict(a['x'])
                    rows.append({'task':task,'seed':seed,'model':'xgboost','partition':p,**eval_metrics(task,frames[p],pp,rr)})
                    save_prediction(task,seed,'xgboost',p,frames[p],pp,rr,a['max_similarity'])
                joblib.dump({'classification':clf,'regression':reg},RESULTS/f'task{task}_xgb_seed_{seed}.joblib',compress=3)
            if task==1 and not any(r['task']==1 and r['seed']==seed and r['model']=='contrast_boost_with_gap_regression' for r in rows):
                tr=arrays['train'];va=arrays['validation']
                reg=XGBRegressor(n_estimators=500,max_depth=4,min_child_weight=20,learning_rate=.04,
                    reg_lambda=10,subsample=.85,colsample_bytree=.85,n_jobs=4,tree_method='hist',
                    early_stopping_rounds=30,random_state=seed)
                reg.fit(tr['x'],tr['target']-tr['anchor'],
                    eval_set=[(va['x'],va['target']-va['anchor'])],verbose=False)
                # The original frozen single booster supplies classification;
                # the independent regressor is a baseline for the continuous
                # endpoint. They are never combined into one score.
                saved=pd.read_csv(OUT/'models_contrast_boost_20261007'/f'predictions_seed_{seed}.csv').set_index('comparison_id')
                f=frames['test'];a=arrays['test'];pp=saved.loc[f.comparison_id,'prediction'].to_numpy()
                rr=reg.predict(a['x'])+a['anchor']
                rows.append({'task':1,'seed':seed,'model':'contrast_boost_with_gap_regression','partition':'test',
                    **eval_metrics(1,f,pp,rr)})
                save_prediction(1,seed,'contrast_boost_with_gap_regression','test',f,pp,rr,a['max_similarity'])
                joblib.dump(reg,RESULTS/f'task1_gap_regression_seed_{seed}.joblib',compress=3)
            for name in kinds:
                attention='attention' in name;empirical_bayes=name.startswith('ebayes')
                distributional=name.startswith('distribution')
                calibrated=name.startswith('calibrated') or distributional
                projected=name.startswith('projected') or empirical_bayes or calibrated
                anchored=name.startswith('anchored') or projected
                used=arrays
                if projected:
                    stop=1024 if task==1 else 512
                    projection=TruncatedSVD(n_components=64,random_state=seed)
                    projection.fit(arrays['train']['x'][:,:stop])
                    projected_scaler=StandardScaler().fit(projection.transform(arrays['train']['x'][:,:stop]))
                    used={p:{**a,'x':np.concatenate([np.clip(projected_scaler.transform(projection.transform(a['x'][:,:stop])),-8,8),a['x'][:,stop:]],1).astype(np.float32)} for p,a in arrays.items()}
                    joblib.dump({'projection':projection,'scaler':projected_scaler,'stop':stop},RESULTS/f'task{task}_projection_seed_{seed}.joblib',compress=3)
                if calibrated:
                    from neural_transition_profiles_20261007 import reference_profiles
                    profile=reference_profiles(frames,obs,seed,task)
                    profile_scaler=StandardScaler().fit(profile['train']['features'])
                    used={p:{**a,'x':np.concatenate([a['x'],np.clip(profile_scaler.transform(profile[p]['features']),-8,8)],1).astype(np.float32),
                             'anchor':profile[p]['mean'],
                             'noise':(profile[p]['features'][:,-1] if task==1 else profile[p]['features'][:,7]).astype(np.float32),
                             'signed_anchor':(profile[p]['features'][:,-2] if task==2 else np.zeros(len(a['y']))).astype(np.float32)} for p,a in used.items()}
                    if distributional and task==1:
                        for p,f in frames.items():
                            margin=np.where(f.gap_A>0,f.min_B_1-f.max_B_2,f.min_B_2-f.max_B_1)
                            used[p]['width_target']=np.maximum(f.target_gap.to_numpy()-margin,0).astype(np.float32)
                    joblib.dump(profile_scaler,RESULTS/f'task{task}_profile_scaler_seed_{seed}.joblib',compress=3)
                    for p in ['validation','test']:
                        pp=profile[p]['risk'];rr=profile[p]['mean']
                        if not any(r['task']==task and r['seed']==seed and r['model']=='reference_calibration_gaussian' and r['partition']==p for r in rows):
                            rows.append({'task':task,'seed':seed,'model':'reference_calibration_gaussian','partition':p,
                                **eval_metrics(task,frames[p],pp,rr)})
                            save_prediction(task,seed,'reference_calibration_gaussian',p,frames[p],pp,rr,arrays[p]['max_similarity'])
                net,pred,hist,epoch=train(used,frames,task,seed,attention,epochs,anchored,projected,empirical_bayes,distributional)
                pd.DataFrame(hist).to_csv(RESULTS/f'task{task}_{name}_history_seed_{seed}.csv',index=False)
                torch.save({'state_dict':net.state_dict(),'dim':used['train']['x'].shape[1],
                            'edge_dim':arrays['train']['edge'].shape[2],'geometry':4 if task==1 else 2,'attention':attention,
                            'anchored':anchored,'projected':projected,'empirical_bayes':empirical_bayes,'calibrated':calibrated,'distributional':distributional,
                            'compact_start':getattr(net,'start',0)},RESULTS/f'task{task}_{name}_seed_{seed}.pt')
                for p,out in pred.items():
                    m=eval_metrics(task,frames[p],out[:,0],out[:,1]);rows.append({'task':task,'seed':seed,'model':name,'partition':p,'best_epoch':epoch,'parameters':sum(v.numel() for v in net.parameters()),**m})
                    save_prediction(task,seed,name,p,frames[p],out[:,0],out[:,1],arrays[p]['max_similarity'])
                print('TASK',task,'SEED',seed,'MODEL',name,'epoch',epoch,'test',rows[-1],flush=True)
                del net
            pd.DataFrame(rows).to_csv(path,index=False);pd.DataFrame(counts).to_csv(RESULTS/'split_counts.csv',index=False)
            print('DONE',task,seed,'seconds',round(time.monotonic()-start,1),flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps({'task1':'Known assay-pair SAR retention, original anchored scaffold splits; robust loss threshold .3',
        'task2':'Known assay-pair molecular residual sensitivity; >=30 training molecules; train-only median pair shift; absolute residual >=.5 pIC50',
        'model':'Single query MLP with learned attention to permitted training references; multitask binary and continuous supervision; no fitted predictor fusion',
        'training_retrieval':'Exclude ALL support rows sharing either query scaffold; 12 nearest label-blind reference candidates',
        'selection':'Validation AP for task1; validation qualifying-pair macro AUROC for task2; patience12',
        'seeds':seeds,'max_epochs':epochs,'source_sha256':digest(OUT/'anchor_transfer_all.parquet'),
        'observation_sha256':digest(OUT/'molecule_assay_observations.parquet'),'code_sha256':digest(Path(__file__)),
        'status':'Exploratory comparison, not prospective validation; old splits have been inspected repeatedly'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--tasks',nargs='+',type=int,default=[1,2]);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--models',nargs='+',default=['direct_neural','reference_attention'])
    a=p.parse_args();main(a.seeds,a.tasks,a.epochs,a.models)
