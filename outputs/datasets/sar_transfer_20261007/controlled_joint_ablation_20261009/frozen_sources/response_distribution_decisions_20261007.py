"""Bayes actions for an absolute Gaussian response under absolute-error loss."""
import numpy as np
from scipy.special import ndtr


def expected_absolute(mu,sd):
    mu=np.abs(np.asarray(mu,dtype=float));sd=np.asarray(sd,dtype=float).clip(1e-8)
    return sd*np.sqrt(2/np.pi)*np.exp(-.5*(mu/sd)**2)+mu*(2*ndtr(mu/sd)-1)


def folded_median(mu,sd):
    mu=np.abs(np.asarray(mu,dtype=float));sd=np.asarray(sd,dtype=float).clip(1e-8)
    lo=np.zeros_like(mu);hi=mu+8*sd
    for _ in range(40):
        mid=(lo+hi)/2;cdf=ndtr((mid-mu)/sd)-ndtr((-mid-mu)/sd)
        lo=np.where(cdf<.5,mid,lo);hi=np.where(cdf>=.5,mid,hi)
    return (lo+hi)/2


def recover_mu_absolute(mean_absolute,variance):
    # Existing GP artifacts saved E|response| and variance rather than mu.
    # This unique inversion uses predictions only, never true query outcomes.
    sd=np.sqrt(np.asarray(variance,dtype=float).clip(1e-10))
    target=np.asarray(mean_absolute,dtype=float);lo=np.zeros_like(target);hi=target.clip(0,None)+8*sd
    for _ in range(40):
        mid=(lo+hi)/2;v=expected_absolute(mid,sd)
        lo=np.where(v<target,mid,lo);hi=np.where(v>=target,mid,hi)
    return (lo+hi)/2,sd
