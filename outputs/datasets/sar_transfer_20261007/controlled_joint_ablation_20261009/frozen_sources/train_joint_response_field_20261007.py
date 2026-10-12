"""Fine-tune one pretrained response distribution on the actual two events.

Point density, SAR-loss probability and molecule-sensitivity probability
share one GP posterior. There is no averaging of predictor outputs.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import ndtr
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,query_arrays,DEVICE,SEEDS
from prepare_sar_transfer_20261007 import OUT,partition,digest
from neural_response_field_20261007 import point_cache
from monotone_response_gp_20261007 import MonotoneResponseGP
from heteroscedastic_response_gp_20261007 import HeteroscedasticResponseGP
from train_monotone_response_20261007 import fast_point_pools,fast_point_batch
from train_known_monotone_response_20261007 import RESULTS as PRETRAIN,infer
from train_known_response_field_20261007 import metric
from audit_response_field_20261007 import separation
from response_distribution_decisions_20261007 import recover_mu_absolute,folded_median
RESULTS=OUT/'models_joint_response_field_20261007';RESULTS.mkdir(exist_ok=True)


def phi(x):return .5*(1+torch.erf(x/np.sqrt(2)))


def probit_event_nll(z,label):
    return -(label*torch.special.log_ndtr(z)+(1-label)*torch.special.log_ndtr(-z)).mean()


def folded_event_nll(mu,sd,label,radius=.5):
    r=mu.abs();lo=(-radius-r)/sd;hi=(radius-r)/sd
    log_lo=torch.special.log_ndtr(lo);log_hi=torch.special.log_ndtr(hi)
    log_inside=log_hi+torch.log(-torch.expm1((log_lo-log_hi).clamp_max(-1e-7)))
    log_outside=torch.logsumexp(torch.stack([log_lo,torch.special.log_ndtr((r-radius)/sd)]),0)
    return -(label*log_outside+(1-label)*log_inside).mean()


def gaussian_folded_median(mu,sd):
    q=torch.sqrt(mu.square()+.4549364231*sd.square())
    for _ in range(5):
        a=(q-mu)/sd;b=(-q-mu)/sd
        cdf=phi(a)-phi(b)
        density=(torch.exp(-.5*a.square())+torch.exp(-.5*b.square()))/(np.sqrt(2*np.pi)*sd)
        q=(q-(cdf-.5)/density.clamp_min(1e-8)).clamp_min(0)
    return q


def contrast_cache(frame,groups,mapping):
    cached=[]
    for pair,f in frame.groupby('pair_id',sort=False):
        if pair not in groups:continue
        pool=groups[pair];sc=pool.scaffold.to_numpy();eligible=[np.flatnonzero((sc!=r.scaffold_1)&(sc!=r.scaffold_2)) for r in f.itertuples()]
        keep=np.array([i for i,ix in enumerate(eligible) if len(ix)>=1])
        if not len(keep):continue
        f=f.iloc[keep];ids,p,gap,target=query_arrays(f,mapping)
        cached.append((ids,p,gap,target,np.array([mapping[m] for m in pool.chemical_identity]),
            ((pool.p_A-6)/2).to_numpy(np.float32),pool.signed_delta.to_numpy(np.float32),[eligible[i] for i in keep],f.binary_loss.to_numpy(np.float32),pair))
    pools={k:[] for k in [1,3,5,10,20,40]}
    for c in cached:
        for k in pools:
            q=np.array([i for i,e in enumerate(c[7]) if len(e)>=k])
            if len(q):pools[k].append((c,q))
    return pools


def contrast_batch(pools,matrix,rng,k,batch=32):
    qi=[];qp=[];gap=[];target=[];si=[];sp=[];sy=[];label=[]
    for _ in range(batch):
        c,q=pools[k][rng.integers(len(pools[k]))];i=int(rng.choice(q));s=rng.choice(c[7][i],k,replace=False)
        qi.append(c[0][i]);qp.append(c[1][i]);gap.append(c[2][i]);target.append(c[3][i]);label.append(c[8][i])
        si.append(c[4][s]);sp.append(c[5][s]);sy.append(c[6][s])
    def t(x):return torch.as_tensor(np.array(x),device=matrix.device)
    return (matrix[t(si)],t(sp),t(sy),matrix[t(qi)],t(qp),t(gap),t(target)),t(label)


def sens_batch(pools,matrix,center,rng,k,batch=32):
    valid=pools[k];si=[];sp=[];sy=[];qi=[];qp=[];target=[];offset=[]
    for _ in range(batch):
        c,q=valid[rng.integers(len(valid))];i=int(rng.choice(q));s=rng.choice(c[3][i],k,replace=False)
        si.append(c[0][s]);sp.append(c[1][s]);sy.append(c[2][s]);qi.append([c[0][i]]);qp.append([c[1][i]]);target.append(c[2][i]);offset.append(center[c[5]])
    def t(x):return torch.as_tensor(np.array(x),device=matrix.device)
    return (matrix[t(si)],t(sp),t(sy),matrix[t(qi)],t(qp),t(target)),t(np.asarray(offset,dtype=np.float32))


def score(model,sar,sens,groups,mapping,matrix):
    a=infer(model,sar,groups,mapping,matrix,1);b=infer(model,sens,groups,mapping,matrix,2)
    m,sd=recover_mu_absolute(b.regression,b.variance);b['expected_absolute']=b.regression;b['regression']=folded_median(m,sd)
    ma=metric(sar,a,1);mb=metric(sens,b,2)
    value=np.mean([ma['AP'],ma['pair_macro_AUROC'],mb['AP'],mb['pair_macro_AUROC']])-.1*(ma['macro_MAE']+mb['macro_MAE'])
    return float(value),ma,mb,a,b


def main(seeds,names,steps):
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=[];path=RESULTS/'metrics.csv'
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.model.isin(names))].to_dict('records')
    for seed in seeds:
        point,train=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False,min_references=1)
        groups={p:g for p,g in train.groupby('pair_id')};center=train.groupby('pair_id').signed_delta.median().to_dict()
        known=train.groupby('pair_id').size();known=set(known[known>=30].index)
        point_pools=fast_point_pools(point,(1,3,5,10,20,40));sens_pools=fast_point_pools([c for c in point if c[5] in known],(1,3,5,10,20,40))
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        sar={p:data.merge(split[split.partition.eq(p)][['comparison_id']],on='comparison_id') for p in ['train','validation','test']}
        sp=obs.scaffold.map(lambda s:partition(s,seed));sens=obs[sp.eq('validation')&obs.pair_id.isin(known)].copy().reset_index(drop=True)
        sens['pair_offset']=sens.pair_id.map(center);sens['residual_abs']=abs(sens.signed_delta-sens.pair_offset);sens['binary_loss']=(sens.residual_abs>=.5).astype(int)
        for part in ['validation','test']:separation(train,sar[part],1,seed,part)
        separation(train,sens,2,seed,'validation');cpools=contrast_cache(sar['train'],groups,mapping)
        print(seed,'contrast query pools',{k:sum(len(q) for _,q in v) for k,v in cpools.items()},flush=True)
        for name in names:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            pretrained='heteroscedastic_monotone' if name=='joint_heteroscedastic' else 'conditional_monotone'
            cp=torch.load(PRETRAIN/f'{pretrained}_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
            matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
            model=(HeteroscedasticResponseGP(cp['dim']) if name=='joint_heteroscedastic' else MonotoneResponseGP(cp['dim'])).to(DEVICE)
            model.load_state_dict(cp['state_dict'])
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.003)
            best,ma,mb,_,_=score(model,sar['validation'],sens,groups,mapping,matrix)
            state=copy.deepcopy(model.state_dict());best_step=0;hist=[{'step':0,'joint_validation_score':best,'SAR_AP':ma['AP'],'sensitivity_AP':mb['AP'],'sensitivity_within_AUROC':mb['pair_macro_AUROC']}]
            for step in range(steps):
                model.train();k=int(rng.choice([1,3,5,10,20,40]));cb,labels=contrast_batch(cpools,matrix,rng,k)
                mean,v=model.contrast(*cb[:-1])
                loss=.25*(v.log()+(cb[-1]-mean).square()/v).mean()+probit_event_nll((.3-mean)/torch.sqrt(v),labels)
                pb,offset=sens_batch(sens_pools,matrix,center,rng,k);mu,var=model.point(*pb[:-1]);sd=var.sqrt();res=mu-offset;truth=abs(pb[-1]-offset)
                loss=loss+folded_event_nll(res,sd,(truth>=.5).float())+.2*F.smooth_l1_loss(gaussian_folded_median(res,sd),truth)
                root=fast_point_batch(point_pools,matrix,rng,k);pm,pv=model.point(*root[:-1])
                loss=loss+.125*(pv.log()+(root[-1]-pm).square()/pv).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0:
                    model.eval();val,ma,mb,_,_=score(model,sar['validation'],sens,groups,mapping,matrix)
                    hist.append({'step':step+1,'loss':float(loss.detach()),'joint_validation_score':val,'SAR_AP':ma['AP'],'sensitivity_AP':mb['AP'],'sensitivity_within_AUROC':mb['pair_macro_AUROC']})
                    print(seed,name,step+1,'score',round(val,4),'AP',round(ma['AP'],4),round(mb['AP'],4),flush=True)
                    if val>best+1e-5:best=val;state=copy.deepcopy(model.state_dict());best_step=step+1
                    if step+1-best_step>=400:break
            model.load_state_dict(state);model.eval();value,ma,mb,a,b=score(model,sar['validation'],sens,groups,mapping,matrix)
            for task,pred,metrics in [(1,a,ma),(2,b,mb)]:
                pred.to_csv(RESULTS/f'task{task}_{name}_validation_seed_{seed}.csv',index=False)
                rows.append({'seed':seed,'task':task,'model':name,'partition':'validation','best_step':best_step,**metrics})
            torch.save({**cp,'state_dict':model.state_dict(),'best_step':best_step,'pretrained_model':pretrained},RESULTS/f'{name}_seed_{seed}.pt')
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{name}_seed_{seed}.csv',index=False);pd.DataFrame(rows).to_csv(path,index=False)
    (RESULTS/'manifest.json').write_text(json.dumps({'training':'One pretrained GP field fine-tuned jointly with original SAR-event BCE+gap NLL,original molecular residual-event BCE+folded-median Huber and root shift NLL',
        'selection':'Mean of4 validation AP/within-pair AUROC scores minus0.1 sum of task macroMAEs;original known-pair validation cohorts only',
        'safeguards':'All support queries exclude entire query scaffold(s);no test scoring;pretrain and fine-tune within same original scaffold split;one posterior,not model fusion',
        'seeds':seeds,'models':names,'steps':steps,'code_sha256':digest(Path(__file__))},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--models',nargs='+',default=['joint_monotone','joint_heteroscedastic']);p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();main(a.seeds,a.models,a.steps)
