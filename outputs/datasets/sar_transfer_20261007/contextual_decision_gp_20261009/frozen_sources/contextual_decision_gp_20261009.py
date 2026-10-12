"""A shared response GP with detached reference-conditioned decision adapters."""
import torch
from torch import nn
from protected_decision_gp_20261009 import ProtectedDecisionGP


def reference_context(xs,ps,ys,xq,pq,offset,variance):
    """Eight reference-only/query-known features, symmetric over support order.

    Accepts batches or a single support set. No query B labels enter this map.
    """
    with torch.no_grad():
        dot=xq[...,:2048]@xs[...,:2048].transpose(-1,-2)
        sim=dot/(xq[...,:2048].sum(-1,keepdim=True)+xs[...,:2048].sum(-1).unsqueeze(-2)-dot).clamp_min(1e-6)
        w=sim.pow(4)+1e-6;w=w/w.sum(-1,keepdim=True)
        localmean=(w*ys.unsqueeze(-2)).sum(-1)
        localstd=(w*(ys.unsqueeze(-2)-localmean.unsqueeze(-1)).square()).sum(-1).sqrt()
        localdistance=(w*(pq.unsqueeze(-1)-ps.unsqueeze(-2)).abs()).sum(-1)
        absolute=(ys-offset.unsqueeze(-1)).abs()
        def broadcast(a):return a.unsqueeze(-1).expand_as(pq)
        features=torch.stack([broadcast(absolute.mean(-1)),broadcast(absolute.std(-1,unbiased=False)),
            broadcast(ys.std(-1,unbiased=False)),pq-broadcast(ps.mean(-1)),sim.max(-1).values,
            localstd,localdistance,variance.clamp_min(1e-5).log()/3],-1)
        return features.clamp(-5,5)


class ContextualDecisionGP(ProtectedDecisionGP):
    def __init__(self,dim,variant='context_linear'):
        super().__init__(dim);self.variant=variant
        def adapter(output):
            if variant=='context_linear':net=nn.Linear(8,output)
            else:net=nn.Sequential(nn.Linear(8,12),nn.SiLU(),nn.Linear(12,output))
            last=net if isinstance(net,nn.Linear) else net[-1]
            nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
            return net
        self.sar_adapter=adapter(2);self.sensitivity_adapter=adapter(2);self.sensitivity_probability_adapter=adapter(1)

    def gp_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith(('sar_risk.','sensitivity_risk.','action_','sar_adapter.','sensitivity_adapter.','sensitivity_probability_adapter.'))]

    def decision_parameters(self):
        gp_ids={id(p) for p in self.gp_parameters()}
        return [p for p in self.parameters() if id(p) not in gp_ids]

    def sar_logits(self,mean,variance,width,features):
        raw=self.sar_adapter(features.detach())
        return self.sar_risk(.3+width-mean.detach(),variance.detach())*torch.exp(.5*torch.tanh(raw[...,1]))+torch.tanh(raw[...,0])

    def sensitivity_action(self,residual,variance,features):
        raw=self.sensitivity_adapter(features.detach())
        base=self.magnitude(residual.detach(),variance.detach())
        return (base*torch.exp(.5*torch.tanh(raw[...,0]))+.25*torch.tanh(raw[...,1])).clamp_min(0)

    def sensitivity_logits(self,q,variance,features):
        # Detach the contextual magnitude, including its scale/bias adapter.
        # The risk adapter has separate parameters from the magnitude adapter.
        raw=self.sensitivity_probability_adapter(features.detach())
        return self.sensitivity_risk(q.detach()-.5,variance.detach())+torch.tanh(raw[...,0])
