"""A single GP with a reference-conditioned monotone assay response mean.

The neural map warps the potency axis. Chemical GP residuals may reverse
rankings; monotonicity constrains only the reference-derived global mean.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from conditional_neural_gp_20261007 import ConditionalGP
from neural_response_field_20261007 import moments


class MonotoneResponseGP(ConditionalGP):
    def __init__(self,dim,conditional=True):
        super().__init__(dim)
        self.conditional=conditional
        self.log_slopes=nn.Parameter(torch.zeros(9))
        self.mean_warp=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,9))
        nn.init.zeros_(self.mean_warp[-1].weight);nn.init.zeros_(self.mean_warp[-1].bias)
        self.register_buffer('knots',torch.linspace(-1.75,1.75,8))
        if not conditional:
            for p in self.context.parameters():p.requires_grad_(False)
            for p in self.mean_warp.parameters():p.requires_grad_(False)

    def warp_parameters(self,ps,ys):
        pm=ps.mean(-1,keepdim=True)
        scale=(ps-pm).square().mean(-1,keepdim=True).sqrt().clamp_min(.25)
        strength=min(1.,max(0.,(ps.shape[-1]-1)/2)) if self.conditional else 0.
        log=self.log_slopes+strength*2*torch.tanh(self.mean_warp(moments(ps,ys)))
        return pm,scale,log.clamp(-2,1.5).exp()

    def integral(self,z,slopes):
        hinge=.15*F.softplus((z[...,None]-self.knots)/.15)
        delta=slopes[...,1:]-slopes[...,:-1]
        return slopes[...,:1]*z+(hinge*delta[...,None,:]).sum(-1)

    def prior_shift(self,p,ps,ys):
        pm,scale,slopes=self.warp_parameters(ps,ys)
        z=(p-pm)/scale
        origin=self.integral(torch.zeros_like(pm),slopes)
        return 2*scale*(self.integral(z,slopes)-origin-z)

    def posterior(self,xs,ps,ys,xq,pq):
        state=self.state(ps,ys);noise=self.noise(state)
        ms=self.prior_shift(ps,ps,ys);mq=self.prior_shift(pq,ps,ys)
        kss=self.kernel(xs,ps,xs,ps,state)+(noise[...,None,None]+1e-4)*torch.eye(ps.shape[-1],device=ps.device)
        kqs=self.kernel(xq,pq,xs,ps,state);kqq=self.kernel(xq,pq,xq,pq,state)
        ch=torch.linalg.cholesky(kss)
        mean=mq+(kqs@torch.cholesky_solve((ys-ms).unsqueeze(-1),ch)).squeeze(-1)
        cov=kqq-kqs@torch.cholesky_solve(kqs.transpose(-1,-2),ch)+noise[...,None,None]*torch.eye(pq.shape[-1],device=ps.device)
        return mean,cov

    def contrast(self,xs,ps,ys,xq,pq,gap_a):
        mean,cov=self.posterior(xs,ps,ys,xq,pq)
        return gap_a+mean[...,0]-mean[...,1],(cov[...,0,0]+cov[...,1,1]-2*cov[...,0,1]).clamp_min(1e-5)

    def point(self,xs,ps,ys,xq,pq):
        mean,cov=self.posterior(xs,ps,ys,xq,pq)
        return mean[...,0],cov[...,0,0].clamp_min(1e-5)
