"""Audit expanded root supervision and conditional response-field invariants."""
import json
import numpy as np
import torch
from meta_gp_sar_20261007 import read, context_components, split_data, SEEDS
from neural_response_field_20261007 import point_cache, NeuralResponseFieldGP
from prepare_sar_transfer_20261007 import partition, OUT


def tokens(frame, columns):
    return {v for c in columns for s in frame[c].dropna().astype(str) for v in s.split(';')}


def separation(train, query, task, seed, part):
    cols=['scaffold_1','scaffold_2'] if task==1 else ['scaffold']
    assert set(train.scaffold).isdisjoint(set().union(*(set(query[c]) for c in cols)))
    for field in ['measurement_id','provenance_equivalence_groups']:
        left=tokens(train,[field+'_A',field+'_B'])
        right=tokens(query,[field+'_'+s+'_'+i for s in ['A','B'] for i in ['1','2']] if task==1 else [field+'_A',field+'_B'])
        assert left.isdisjoint(right),(seed,part,task,field)


def main():
    torch.set_num_threads(2)
    obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet')
    roots=context_components(obs);mapping={m:i for i,m in enumerate(obs.chemical_identity.unique())}
    rows=[]
    for seed in SEEDS:
        parts,*_=split_data(obs,data,seed,roots)
        cache,train=point_cache(obs,seed,roots,mapping)
        for part in ['validation','test']:
            separation(train,parts[part],1,seed,part)
            assert set(train.pair_id.map(roots)).isdisjoint(set(parts[part].pair_id.map(roots)))
        _,known=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False)
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        for part in ['validation','test']:
            sar=data.merge(split[split.partition.eq(part)][['comparison_id']],on='comparison_id')
            sens=obs[obs.scaffold.map(lambda s:partition(s,seed)).eq(part)]
            separation(known,sar,1,seed,part);separation(known,sens,2,seed,part)
        rows.append({'seed':seed,'strict_train_rows':len(train),'strict_point_pairs':len(cache),
                     'known_pair_train_rows':len(known),'validation_test_label_and_scaffold_separation':True})
    torch.manual_seed(23);net=NeuralResponseFieldGP(2059).double().eval()
    with torch.no_grad():
        net.mean_response[-1].weight.normal_(0,.05);net.context[-1].weight.normal_(0,.05)
        x=torch.randint(0,2,(3,9,2059)).double();p=torch.randn(3,9,dtype=torch.double);y=torch.randn(3,6,dtype=torch.double)
        mu,cov=net.posterior(x[:,:6],p[:,:6],y,x[:,6:],p[:,6:])
        ix=torch.randperm(6);mu2,cov2=net.posterior(x[:,:6][:,ix],p[:,:6][:,ix],y[:,ix],x[:,6:],p[:,6:])
        permutation_error=float(max((mu-mu2).abs().max(),(cov-cov2).abs().max()));assert permutation_error<1e-8
        eig=float(torch.linalg.eigvalsh(cov).min());assert eig>0
        gap=torch.tensor([.7,.9,1.],dtype=torch.double)
        a,v=net.contrast(x[:,:6],p[:,:6],y,x[:,6:8],p[:,6:8],gap)
        b,w=net.contrast(x[:,:6],p[:,:6],y,x[:,6:8].flip(1),p[:,6:8].flip(1),-gap)
        assert torch.allclose(a,-b,atol=1e-8) and torch.allclose(v,w,atol=1e-8)
    result={'splits':rows,'numerical':{'nonzero_neural_heads':True,'reference_permutation_error':permutation_error,
        'minimum_posterior_eigenvalue':eig,'contrast_swap_antisymmetry':True}}
    (OUT/'models_conditional_neural_gp_20261007'/'expanded_supervision_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
