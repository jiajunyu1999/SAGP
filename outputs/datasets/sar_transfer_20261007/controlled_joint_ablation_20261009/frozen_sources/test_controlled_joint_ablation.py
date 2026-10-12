"""Mechanistic and paired-bootstrap invariants of the controlled experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from controlled_joint_ablation_20261009 import ControlledGP
from summarize_controlled_joint_ablation_20261009 import ap_cache,weighted_ap,contrasts,prepare


def test_covariance_ablation_changes_only_contrast_variance():
    joint=ControlledGP(2059,'joint');independent=ControlledGP(2059,'no_contrast_covariance')
    args=[torch.tensor([v]) for v in [.2,-.3,1.,2.,.4,1.5]]
    m,v=joint.contrast_moments(*args);mi,vi=independent.contrast_moments(*args)
    assert torch.equal(m,mi)
    assert torch.allclose(v,torch.tensor([2.2]))
    assert torch.allclose(vi,torch.tensor([3.]))


def test_chemical_kernel_ablation_is_invariant_to_query_structure():
    torch.manual_seed(17);m=ControlledGP(2059,'no_molecular_residual')
    x=torch.rand(1,3,2059);y=torch.rand(1,2,2059)
    p=torch.randn(1,3);q=torch.randn(1,2);state=torch.zeros(1,4)
    assert torch.equal(m.kernel(x,p,y,q,state),m.kernel(torch.zeros_like(x),p,torch.ones_like(y),q,state))


def test_nonmonotone_ablation_allows_negative_slopes():
    m=ControlledGP(2059,'no_monotone_mean')
    with torch.no_grad():m.log_slopes.fill_(-2)
    assert (m.warp_parameters(torch.zeros(1,3),torch.zeros(1,3))[2]<0).all()


def test_shared_heads_start_at_identical_predictive_moments():
    joint=ControlledGP(2059,'joint');head=ControlledGP(2059,'shared_multihead')
    assert sum(p.numel() for p in head.parameters())-sum(p.numel() for p in joint.parameters())==58
    args=[torch.tensor([v]) for v in [.2,-.3,1.,2.,.4,1.5]]
    for a,b in zip(joint.contrast_moments(*args),head.contrast_moments(*args)):assert torch.equal(a,b)


def test_cached_weighted_ap_matches_sklearn_with_ties_and_zero_blocks():
    y=np.array([0,1,0,1,1,0]);p=np.array([.8,.8,.5,.5,.2,.1]);c=np.array([0,1,0,1,2,2])
    for w in [np.ones(3),np.array([0.,2.,1.]),np.array([3.,0.,2.])]:
        assert np.isclose(weighted_ap(ap_cache(y,p,c),w),average_precision_score(y,p,sample_weight=w[c]))


def test_bootstrap_does_not_drop_unevaluable_seeds():
    aa=ap_cache(np.array([1,0]),np.array([.8,.2]),np.array([0,1]))
    first=(aa,aa,np.array([0]),np.array([1.]),np.array([0,1]),np.array([0.,0.]))
    second=(aa,aa,np.array([1]),np.array([0.]),np.array([0,1]),np.array([0.,0.]))
    out=contrasts([first,second],np.array([[1.,1.],[0.,2.]]))
    assert out[0,1]==.5
    assert np.isnan(out[1,1])


def test_pairing_accepts_exact_same_labels_in_different_integer_dtypes():
    import pandas as pd
    a=pd.DataFrame({'comparison_id':['x','y'],'pair_id':['p','p'],'binary_loss':[0,1],
                    'target':[0.,1.],'regression':[.1,.9],'prediction':[.2,.8]})
    b=a.copy();b['binary_loss']=b.binary_loss.astype(np.int8)
    packed=prepare(a,b,{'p':'root'},{'root':0})
    assert np.array_equal(packed[-1],np.array([0.]))
