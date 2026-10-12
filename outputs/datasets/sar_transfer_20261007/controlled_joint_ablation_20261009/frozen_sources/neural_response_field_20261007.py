"""Neural conditional signed-response mean inside one GP.

Extra root-data episodes supervise an individual molecule's A-to-B shift,
without requiring an analogue pair or a large assay-A gap. Queries and their
whole scaffolds are excluded from their reference pools. Test contexts and
query B outcomes remain outside training.
"""
import numpy as np
import torch
from torch import nn
from conditional_neural_gp_20261007 import ConditionalGP
from prepare_sar_transfer_20261007 import partition


def moments(p,y):
    pm=p.mean(-1);ym=y.mean(-1);dp=p-pm[...,None];dy=y-ym[...,None]
    vp=dp.square().mean(-1);vy=dy.square().mean(-1);cov=(dp*dy).mean(-1)
    slope=cov/(vp+.05);res=(dy-slope[...,None]*dp).square().mean(-1).sqrt()
    return torch.stack([pm,ym,vp.sqrt(),vy.sqrt(),cov,slope,res,torch.full_like(pm,np.log1p(p.shape[-1])/3)],-1).clamp(-8,8)


class NeuralResponseFieldGP(ConditionalGP):
    def __init__(self,dim):
        super().__init__(dim)
        self.mean_molecule=nn.Sequential(nn.Linear(523,64),nn.SiLU(),nn.Linear(64,32))
        self.mean_condition=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,32))
        self.mean_response=nn.Sequential(nn.Linear(97,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.mean_response[-1].weight);nn.init.zeros_(self.mean_response[-1].bias)
    def raw_mean(self,x,p,summary):
        folded=x[...,:2048].reshape(*x.shape[:-1],4,512).amax(-2)
        mol=self.mean_molecule(torch.cat([folded,x[...,2048:]],-1))
        cond=self.mean_condition(summary)[...,None,:].expand(*mol.shape)
        return self.mean_response(torch.cat([mol,cond,mol*cond,p[...,None]],-1)).squeeze(-1)
    def posterior(self,xs,ps,ys,xq,pq):
        state=self.state(ps,ys);noise=self.noise(state);pm=ps.mean(-1,keepdim=True)
        summary=moments(ps,ys)
        us=self.raw_mean(xs,ps-pm,summary);uq=self.raw_mean(xq,pq-pm,summary);center=us.mean(-1,keepdim=True)
        prior_s=state[...,3,None]*(ps-pm)+us-center
        prior_q=state[...,3,None]*(pq-pm)+uq-center
        kss=self.kernel(xs,ps,xs,ps,state)+(noise[...,None,None]+1e-4)*torch.eye(ps.shape[-1],device=ps.device)
        kqs=self.kernel(xq,pq,xs,ps,state);kqq=self.kernel(xq,pq,xq,pq,state)
        ch=torch.linalg.cholesky(kss)
        alpha=torch.cholesky_solve((ys-prior_s).unsqueeze(-1),ch)
        mean=prior_q+(kqs@alpha).squeeze(-1)
        cov=kqq-kqs@torch.cholesky_solve(kqs.transpose(-1,-2),ch)+noise[...,None,None]*torch.eye(pq.shape[-1],device=ps.device)
        return mean,cov
    def contrast(self,xs,ps,ys,xq,pq,gap_a):
        mean,cov=self.posterior(xs,ps,ys,xq,pq)
        return gap_a+mean[...,0]-mean[...,1],(cov[...,0,0]+cov[...,1,1]-2*cov[...,0,1]).clamp_min(1e-5)
    def point(self,xs,ps,ys,xq,pq):
        mean,cov=self.posterior(xs,ps,ys,xq,pq)
        return mean[...,0],cov[...,0,0].clamp_min(1e-5)


def point_cache(obs,seed,roots,mapping,strict_context=True,scaffold_salt=True,min_references=3):
    cp={c:partition('ctx|'+c,seed) for c in set(roots.values())};sp={s:partition(('scf|' if scaffold_salt else '')+s,seed) for s in obs.scaffold.unique()}
    eligible_context=obs.pair_id.map(roots).map(cp).eq('train') if strict_context else np.ones(len(obs),bool)
    train=obs[eligible_context&obs.scaffold.map(sp).eq('train')]
    cached=[]
    for pair,g in train.groupby('pair_id'):
        scaffold=g.scaffold.to_numpy();eligible=[np.flatnonzero(scaffold!=s) for s in scaffold]
        queries=np.array([i for i,e in enumerate(eligible) if len(e)>=min_references])
        if not len(queries):continue
        cached.append((np.array([mapping[m] for m in g.chemical_identity]),((g.p_A-6)/2).to_numpy(np.float32),
                       (g.p_B-g.p_A).to_numpy(np.float32),eligible,queries,pair))
    return cached,train


def point_batch(cached,matrix,rng,k,batch=32):
    valid=[]
    for c in cached:
        q=np.array([i for i in c[4] if len(c[3][i])>=k])
        if len(q):valid.append((c,q))
    assert valid
    si=[];sp=[];sy=[];qi=[];qp=[];target=[]
    for _ in range(batch):
        c,q=valid[rng.integers(len(valid))];i=int(rng.choice(q));s=rng.choice(c[3][i],k,replace=False)
        si.append(c[0][s]);sp.append(c[1][s]);sy.append(c[2][s]);qi.append([c[0][i]]);qp.append([c[1][i]]);target.append(c[2][i])
    def t(x):return torch.as_tensor(np.array(x),device=matrix.device)
    return matrix[t(si)],t(sp),t(sy),matrix[t(qi)],t(qp),t(target)


def conditional_point(model,xs,ps,ys,xq,pq):
    if hasattr(model,'point'):return model.point(xs,ps,ys,xq,pq)
    state=model.state(ps,ys);noise=model.noise(state);pm=ps.mean(-1,keepdim=True)
    prior_s=state[...,3,None]*(ps-pm);prior_q=state[...,3,None]*(pq-pm)
    kss=model.kernel(xs,ps,xs,ps,state)+(noise[...,None,None]+1e-4)*torch.eye(ps.shape[-1],device=ps.device)
    kqs=model.kernel(xq,pq,xs,ps,state);kqq=model.kernel(xq,pq,xq,pq,state);ch=torch.linalg.cholesky(kss)
    mean=prior_q+(kqs@torch.cholesky_solve((ys-prior_s).unsqueeze(-1),ch)).squeeze(-1)
    cov=kqq-kqs@torch.cholesky_solve(kqs.transpose(-1,-2),ch)+noise[...,None,None]*torch.eye(pq.shape[-1],device=ps.device)
    return mean[...,0],cov[...,0,0].clamp_min(1e-5)
