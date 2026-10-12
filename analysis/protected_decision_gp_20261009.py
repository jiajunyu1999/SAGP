"""One chemical GP with protected density learning and endpoint decision rules.

Binary gradients update only decision calibration, not GP mean/covariance.
The sensitivity action has its own magnitude objective. An optional projection
removes negative raw-gradient alignment with the continuous GP density loss.
No predictions from independently fitted models are combined.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from monotone_response_gp_20261007 import MonotoneResponseGP
from train_joint_response_field_20261007 import gaussian_folded_median


class RiskRule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(0.))
        self.log_temperature=nn.Parameter(torch.tensor(np.log(1.702),dtype=torch.float32))
        self.raw_variance=nn.Parameter(torch.tensor(np.log(np.expm1(1.)),dtype=torch.float32))
        self.log_floor=nn.Parameter(torch.tensor(np.log(.05),dtype=torch.float32))

    def forward(self,margin,variance):
        denominator=torch.sqrt(F.softplus(self.raw_variance)*variance.clamp_min(1e-5)+self.log_floor.clamp(-5,1).exp().square())
        return self.bias+self.log_temperature.clamp(-3,3).exp()*margin/denominator


class ProtectedDecisionGP(MonotoneResponseGP):
    def __init__(self,dim):
        super().__init__(dim)
        self.sar_risk=RiskRule();self.sensitivity_risk=RiskRule()
        self.action_log_scale=nn.Parameter(torch.tensor(0.))
        self.action_raw_variance=nn.Parameter(torch.tensor(np.log(np.expm1(1.)),dtype=torch.float32))
        self.action_bias=nn.Parameter(torch.tensor(0.))

    def magnitude(self,residual,variance):
        sd=torch.sqrt(F.softplus(self.action_raw_variance)*variance.clamp_min(1e-5))
        q=gaussian_folded_median(residual,sd)
        return (self.action_log_scale.clamp(-2,2).exp()*q+self.action_bias).clamp_min(0)

    def gp_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith(('sar_risk.','sensitivity_risk.','action_'))]

    def decision_parameters(self):
        return [p for n,p in self.named_parameters() if n.startswith(('sar_risk.','sensitivity_risk.','action_'))]


def project_auxiliary(anchor,auxiliary):
    """Global Euclidean projection; preserves nonnegative raw alignment.

    This is a statement about gradients, not a monotone-loss guarantee under
    AdamW, which also applies preconditioning and weight decay.
    """
    dot=sum((a*b).sum() for a,b in zip(anchor,auxiliary))
    norm=sum(a.square().sum() for a in anchor)
    coefficient=torch.minimum(dot,torch.zeros_like(dot))/(norm+1e-12)
    return [b-coefficient*a for a,b in zip(anchor,auxiliary)],float(dot.detach())


def protected_classification_loss(model,mean,var,width,labels,q,svar,slabels):
    # Width comes only from sampled historical references, matching inference.
    l1=F.binary_cross_entropy_with_logits(model.sar_risk(.3+width-mean.detach(),var.detach()),labels)
    l2=F.binary_cross_entropy_with_logits(model.sensitivity_risk(q.detach()-.5,svar.detach()),slabels)
    return l1+l2
