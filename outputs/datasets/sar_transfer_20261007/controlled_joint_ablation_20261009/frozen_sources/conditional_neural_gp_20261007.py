"""Outcome-conditioned neural transition GP and myopic reference design.

A permutation-invariant neural context network amortises transition-specific
covariance/noise and a linear prior mean from permitted reference observations.
One GP posterior predicts a JOINT query contrast. Hyperparameters depend only
on selected reference B labels, never on candidate/query B outcomes.
"""
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr
from sklearn.metrics import average_precision_score
from meta_gp_sar_20261007 import (MetaGP,RESULTS as BASE,DEVICE,read,context_components,split_data,
    molecule_matrix,train_samples,sample_batch,query_arrays,stable_seed,summarize,SEEDS)
from prepare_sar_transfer_20261007 import OUT,digest
RESULTS=OUT/'models_conditional_neural_gp_20261007';RESULTS.mkdir(exist_ok=True)

class ConditionalGP(MetaGP):
    def __init__(self,dim):
        super().__init__(dim,'tanimoto')
        self.context=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,4))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
    def state(self,p,y):
        pm=p.mean(-1);ym=y.mean(-1);dp=p-pm[...,None];dy=y-ym[...,None]
        vp=dp.square().mean(-1);vy=dy.square().mean(-1);cov=(dp*dy).mean(-1)
        slope=cov/(vp+.05);res=(dy-slope[...,None]*dp).square().mean(-1).sqrt()
        features=torch.stack([pm,ym,vp.sqrt(),vy.sqrt(),cov,slope,res,torch.full_like(pm,math.log1p(p.shape[-1])/3)],-1)
        raw=self.context(features.clamp(-8,8))
        # The network is trained on K>=3. Single-reference initiation falls
        # back to the trained global process instead of inventing moments.
        strength=min(1.,max(0.,(p.shape[-1]-1)/2))
        return 2*torch.tanh(raw)*strength
    def noise(self,state=None):
        log=self.log_noise.clamp(-4,1)
        if state is not None:log=log+state[...,2]
        return log.clamp(-5,2).exp().square()+1e-5
    def kernel(self,x,p,y,q,state=None):
        offset=self.log_offset.clamp(-3,3).exp().square()
        slope=self.log_slope.clamp(-4,2).exp().square()
        amp=self.log_amp.clamp(-4,2).exp().square()
        if state is not None:
            slope=slope*state[...,0].exp();amp=amp*state[...,1].exp()
        dot=x[...,:2048]@y[...,:2048].transpose(-1,-2)
        tani=dot/(x[...,:2048].sum(-1,keepdim=True)+y[...,:2048].sum(-1).unsqueeze(-2)-dot).clamp_min(1e-6)
        return offset+slope[...,None,None]*p.unsqueeze(-1)*q.unsqueeze(-2)+amp[...,None,None]*tani
    def contrast(self,xs,ps,ys,xq,pq,gap_a):
        state=self.state(ps,ys);noise=self.noise(state)
        pm=ps.mean(-1,keepdim=True);prior_s=state[...,3,None]*(ps-pm)
        prior_q=state[...,3,None]*(pq-pm)
        kss=self.kernel(xs,ps,xs,ps,state)+(noise[...,None,None]+1e-4)*torch.eye(ps.shape[-1],device=ps.device)
        kqs=self.kernel(xq,pq,xs,ps,state);kqq=self.kernel(xq,pq,xq,pq,state)
        chol=torch.linalg.cholesky(kss);alpha=torch.cholesky_solve((ys-prior_s).unsqueeze(-1),chol).squeeze(-1)
        cross=kqs[...,0,:]-kqs[...,1,:]
        mean=gap_a+prior_q[...,0]-prior_q[...,1]+(cross*alpha).sum(-1)
        var=kqq[...,0,0]+kqq[...,1,1]-2*kqq[...,0,1]-(cross*torch.cholesky_solve(cross.unsqueeze(-1),chol).squeeze(-1)).sum(-1)+2*noise
        return mean,var.clamp_min(1e-5)


def evaluate(model,episodes,groups,mapping,matrix,seed,k=5,draw=0):
    output=[];model.eval()
    with torch.no_grad():
        for pair,f in episodes.items():
            if hasattr(model,'context_lookup'):
                model.set_context(torch.as_tensor(model.context_lookup[pair],device=DEVICE))
            pool=groups[pair];rng=np.random.default_rng(stable_seed(seed,pair,'random') if draw==0 else stable_seed(seed,pair,'random',draw))
            s=pool.iloc[rng.permutation(len(pool))[:k]];ids,p,gap,target=query_arrays(f,mapping)
            si=torch.tensor([mapping[m] for m in s.chemical_identity],device=DEVICE)
            xs=matrix[si][None].expand(len(f),-1,-1)
            ps=torch.tensor(((s.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)[None].expand(len(f),-1)
            ys=torch.tensor((s.p_B-s.p_A).to_numpy(np.float32),device=DEVICE)[None].expand(len(f),-1)
            mean,var=model.contrast(xs,ps,ys,matrix[torch.tensor(ids,device=DEVICE)],torch.tensor(p,device=DEVICE),torch.tensor(gap,device=DEVICE))
            output.append(prediction_frame(f,target,mean.cpu().numpy(),var.cpu().numpy()))
    return pd.concat(output,ignore_index=True)


def prediction_frame(frame,target,mean,var):
    out=frame[['comparison_id','pair_id','target_chembl_id','binary_loss']].copy()
    out['gap_B_target']=target;out['gap_B_mean']=mean;out['gap_B_variance']=var
    out['prediction']=ndtr((.3-mean)/np.sqrt(var));out['nll']=.5*(np.log(2*np.pi*var)+(target-mean)**2/var)
    out['halfwidth_90']=1.644854*np.sqrt(var);out['AE']=abs(target-mean)
    return out


def fit(seed,kind,matrix,cached,episodes,groups,mapping,steps,dim,point_data=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if kind=='conditional_mean_point':
        from neural_response_field_20261007 import NeuralResponseFieldGP
        model=NeuralResponseFieldGP(dim).to(DEVICE)
    else:model=ConditionalGP(dim).to(DEVICE)
    base=torch.load(BASE/f'tanimoto_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(base['state_dict'],strict=False)
    optimizer=torch.optim.AdamW([
        {'params':list(model.context.parameters()),'lr':.001,'weight_decay':.003},
        {'params':[p for n,p in model.named_parameters() if n.startswith('mean_')],'lr':.0005,'weight_decay':.003},
        {'params':[p for n,p in model.named_parameters() if not n.startswith(('context.','mean_'))],'lr':.003,'weight_decay':0.}])
    best=-1;best_step=0;state=None;hist=[]
    for step in range(steps):
        k=int(rng.choice([3,5,10]));model.train()
        # Query/class targets are drawn in parallel with the original sampler.
        # Reset RNG to recover exactly the sampled episode/query indices.
        before=copy.deepcopy(rng.bit_generator.state)
        if kind in ['conditional_root_pairrank','conditional_root_pairrank_strong']:
            batch,labels=sample_pairrank_batch(cached,matrix,rng,k)
        else:
            batch=sample_batch(cached,matrix,rng,k)
            after=copy.deepcopy(rng.bit_generator.state);rng.bit_generator.state=before
            labels=[]
            for _ in range(32):
                c=cached[rng.integers(len(cached))];i=rng.integers(len(c[0]));rng.choice(c[7][i],k,replace=False)
                labels.append(c[8][i])
            rng.bit_generator.state=after
        mean,var=model.contrast(*batch[:-1]);target=batch[-1]
        loss=.5*(var.log()+(target-mean).square()/var).mean()
        if kind in ['conditional_multitask','conditional_mean_point','conditional_root_point','conditional_root_pairrank','conditional_root_pairrank_strong']:
            prob=.5*(1+torch.erf((.3-mean)/torch.sqrt(2*var)))
            loss=loss+.5*F.binary_cross_entropy(prob.clamp(1e-5,1-1e-5),torch.tensor(labels,device=DEVICE,dtype=torch.float32))
        if kind in ['conditional_root_pairrank','conditional_root_pairrank_strong']:
            yy=torch.tensor(labels,device=DEVICE,dtype=torch.float32).reshape(-1,4)
            pp=prob.reshape(-1,4)
            pos=yy[:,:,None]>.5;neg=yy[:,None,:]<.5
            pair_logits=(pp[:,:,None]-pp[:,None,:])/.15
            rank_weight=2.0 if kind=='conditional_root_pairrank_strong' else .5
            loss=loss+rank_weight*F.softplus(-pair_logits[pos&neg]).mean()
        if point_data is not None:
            from neural_response_field_20261007 import point_batch,conditional_point
            pb=point_batch(point_data,matrix,rng,k)
            mu,v=conditional_point(model,*pb[:-1]);point_loss=.5*(v.log()+(pb[-1]-mu).square()/v).mean()
            loss=loss+.5*point_loss
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if (step+1)%100==0:
            predictions=[evaluate(model,episodes['validation'],groups,mapping,matrix,seed,5,d) for d in range(3)]
            val_ap=float(np.mean([average_precision_score(p.binary_loss,p.prediction) for p in predictions]))
            val_pair_auc=float(np.mean([summarize(p)['pair_macro_AUROC'] for p in predictions]))
            val_nll=float(np.mean([summarize(p)['macro_NLL'] for p in predictions]))
            hist.append({'step':step+1,'loss':float(loss.detach()),'val_AP_three_draws':val_ap,
                         'val_pair_macro_AUROC_three_draws':val_pair_auc,'val_macro_NLL_three_draws':val_nll})
            selection_score=val_pair_auc if kind=='conditional_root_pairrank_strong' else val_ap
            if selection_score>best+1e-5:best=selection_score;best_step=step+1;state=copy.deepcopy(model.state_dict())
            if step+1-best_step>=500:break
    model.load_state_dict(state);model.eval();return model,hist,best_step


def sample_pairrank_batch(cached,matrix,rng,k,pairs_per_batch=8):
    """Sample balanced query blocks from one assay pair and shared references."""
    eligible=[]
    for c in cached:
        labels=np.asarray(c[8],dtype=np.int8)
        pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
        if len(pos)>=2 and len(neg)>=2:eligible.append((c,pos,neg))
    qi=[];qp=[];ga=[];target=[];si=[];sp=[];sy=[];yy=[]
    for _ in range(pairs_per_batch):
        chosen=None
        for attempt in range(50):
            c,pos,neg=eligible[int(rng.integers(len(eligible)))]
            ix=np.r_[rng.choice(pos,2,replace=False),rng.choice(neg,2,replace=False)]
            common=np.asarray(c[7][ix[0]],dtype=int)
            for j in ix[1:]:common=np.intersect1d(common,c[7][j],assume_unique=False)
            if len(common)>=k:
                chosen=(c,ix,rng.choice(common,k,replace=False));break
        if chosen is None:raise RuntimeError('Could not form scaffold-disjoint pair-ranking episodes')
        c,ix,s=chosen
        for i in ix:
            qi.append(c[0][i]);qp.append(c[1][i]);ga.append(c[2][i]);target.append(c[3][i])
            si.append(c[4][s]);sp.append(c[5][s]);sy.append(c[6][s]);yy.append(c[8][i])
    def tensor(a):return torch.as_tensor(np.asarray(a),device=matrix.device)
    batch=(matrix[tensor(si)],tensor(sp),tensor(sy),matrix[tensor(qi)],tensor(qp),tensor(ga),tensor(target))
    return batch,np.asarray(yy,dtype=np.float32)


def design(model,episodes,groups,mapping,bits,matrix,seed,draw):
    outputs={(policy,k):[] for policy in ['random','diversity','contrast_design'] for k in [3,5,10]};selections=[]
    for pair,frame in episodes.items():
        if hasattr(model,'context_lookup'):
            model.set_context(torch.as_tensor(model.context_lookup[pair],device=DEVICE))
        original=groups[pair];rng=np.random.default_rng(stable_seed(seed,pair,'random') if draw==0 else stable_seed(seed,pair,'random',draw))
        pool=original.iloc[rng.permutation(len(original))[:100]].reset_index(drop=True)
        qids,qp,gap,target=query_arrays(frame,mapping);unique=np.unique(qids);lookup={m:len(pool)+i for i,m in enumerate(unique)}
        qpmap={int(m):float(p) for ii,pp in zip(qids,qp) for m,p in zip(ii,pp)}
        ids=np.concatenate([[mapping[m] for m in pool.chemical_identity],unique])
        p=np.concatenate([((pool.p_A-6)/2).to_numpy(np.float32),[qpmap[int(m)] for m in unique]]).astype(np.float32)
        x=matrix[torch.tensor(ids,device=DEVICE)];pt=torch.tensor(p,device=DEVICE)
        qs=np.array([lookup[m] for m in qids[:,0]]);qw=np.array([lookup[m] for m in qids[:,1]])
        for policy in ['random','diversity','contrast_design']:
            selected=[0]
            while len(selected)<=10:
                # Read only selected B outcomes, after the measurement choice.
                y=(pool.iloc[selected].p_B-pool.iloc[selected].p_A).to_numpy(np.float32)
                with torch.no_grad():
                    s=torch.tensor(selected,device=DEVICE);yt=torch.tensor(y,device=DEVICE)
                    ctx=model.state(pt[s],yt);kernel=model.kernel(x,pt,x,pt,ctx).cpu().numpy().astype(np.float64)
                    noise=float(model.noise(ctx));slope=float(ctx[3]);ref_center=float(pt[s].mean())
                    if hasattr(model,'observation_noise'):
                        noise_vector=model.observation_noise(x,pt,pt[s],yt,ctx).cpu().numpy().astype(np.float64)
                    else:noise_vector=np.full(len(p),noise)
                observed=kernel[np.ix_(selected,selected)]+np.diag(noise_vector[selected]+1e-4)
                cross=kernel[:,selected];solve=np.linalg.solve(observed,cross.T)
                post=kernel-cross@solve
                if len(selected) in [3,5,10]:
                    prior=slope*(p-ref_center)
                    if hasattr(model,'prior_shift'):
                        with torch.no_grad():
                            prior=model.prior_shift(pt,pt[s],yt).cpu().numpy()
                    elif hasattr(model,'raw_mean'):
                        from neural_response_field_20261007 import moments
                        with torch.no_grad():
                            raw=model.raw_mean(x,pt-float(ref_center),moments(pt[s],yt)).cpu().numpy()
                        prior=prior+raw-raw[selected].mean()
                    centered=y-prior[selected]
                    shift=prior+cross@np.linalg.solve(observed,centered)
                    mean=gap+shift[qs]-shift[qw];var=(post[qs,qs]+post[qw,qw]-2*post[qs,qw]+noise_vector[qs]+noise_vector[qw]).clip(1e-5)
                    outputs[policy,len(selected)].append(prediction_frame(frame,target,mean,var))
                if len(selected)==10:break
                if policy=='random':score=-np.arange(len(pool),dtype=float)
                elif policy=='diversity':
                    fp=bits[ids[:len(pool)]];dot=fp@fp[selected].T
                    similarity=dot/(fp.sum(1)[:,None]+fp[selected].sum(1)[None,:]-dot).clip(1e-6)
                    score=1-similarity.max(1)
                else:
                    cov=post[qs,:len(pool)]-post[qw,:len(pool)]
                    score=(cov**2/(np.diag(post)[:len(pool)]+noise_vector[:len(pool)]+1e-4).clip(1e-8)[None,:]).mean(0)
                score[selected]=-np.inf;selected.append(int(np.argmax(score)))
            for rank,ix in enumerate(selected):selections.append({'seed':seed,'draw':draw,'pair_id':pair,'policy':policy,'step':rank+1,'chemical_identity':pool.iloc[ix].chemical_identity})
    return {key:pd.concat(value,ignore_index=True) for key,value in outputs.items()},selections


def main(seeds,kinds,steps,validation_only=False):
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=[];selection=[]
    path=RESULTS/'metrics.csv'
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.model.isin(kinds))].to_dict('records')
    for seed in seeds:
        parts,episodes,groups,trainobs,counts=split_data(obs,data,seed,roots)
        base=torch.load(BASE/f'tanimoto_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-base['desc_mean'])/base['desc_std'],-8,8)],1),device=DEVICE)
        cached=train_samples(episodes['train'],groups,mapping)
        # Match cache order to the episode iteration order for class labels.
        cached=[c+(f.binary_loss.to_numpy(np.float32),) for c,f in zip(cached,episodes['train'].values())]
        point_data=None
        if any(kind in ['conditional_mean_point','conditional_root_point','conditional_root_pairrank','conditional_root_pairrank_strong'] for kind in kinds):
            from neural_response_field_20261007 import point_cache
            point_data,extra_train=point_cache(obs,seed,roots,mapping)
            assert set(extra_train.scaffold).isdisjoint(set(parts['test'].scaffold_1)|set(parts['test'].scaffold_2))
            assert set(extra_train.pair_id.map(roots)).isdisjoint(set(parts['test'].pair_id.map(roots)))
            print(seed,'extra point episodes',len(point_data),'root training rows',len(extra_train),flush=True)
        for kind in kinds:
            start=time.monotonic();model,hist,step=fit(seed,kind,matrix,cached,episodes,groups,mapping,steps,matrix.shape[1],
                point_data if kind in ['conditional_mean_point','conditional_root_point','conditional_root_pairrank','conditional_root_pairrank_strong'] else None)
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            torch.save({'state_dict':model.state_dict(),'dim':matrix.shape[1],'desc_mean':base['desc_mean'],'desc_std':base['desc_std'],'best_step':step},RESULTS/f'{kind}_seed_{seed}.pt')
            if validation_only:
                validation=[]
                for draw in range(3):
                    pred=evaluate(model,episodes['validation'],groups,mapping,matrix,seed,5,draw)
                    validation.append({'seed':seed,'draw':draw,'model':kind,**summarize(pred)})
                vp=RESULTS/'validation_pairrank_pilot.csv'
                oldv=pd.read_csv(vp) if vp.exists() else pd.DataFrame()
                oldv=oldv[~(oldv.seed.eq(seed)&oldv.model.eq(kind))] if len(oldv) else oldv
                pd.concat([oldv,pd.DataFrame(validation)],ignore_index=True).to_csv(vp,index=False)
                print(seed,kind,'validation only','best_step',step,
                    pd.DataFrame(validation)[['AP','pair_macro_AUROC','macro_MAE']].mean().round(4).to_dict(),flush=True)
                continue
            for draw in range(3):
                outputs,chosen=design(model,episodes['test'],groups,mapping,bits,matrix,seed,draw)
                selection.extend([{'model':kind,**r} for r in chosen])
                for (policy,k),pred in outputs.items():
                    pred.to_csv(RESULTS/f'predictions_{kind}_{policy}_k{k}_seed_{seed}_draw{draw}.csv',index=False)
                    rows.append({'seed':seed,'draw':draw,'model':kind,'policy':policy,'support_k':k,'best_step':step,**summarize(pred)})
            pd.DataFrame(rows).to_csv(path,index=False);pd.DataFrame(selection).to_csv(RESULTS/'reference_selections.csv',index=False)
            print(seed,kind,'step',step,'seconds',round(time.monotonic()-start,1),
                pd.DataFrame(rows).query("seed==@seed and model==@kind and support_k==5").groupby('policy')[['AP','AUROC','macro_MAE']].mean().round(4).to_dict(),flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps({'task':'Same strict source/context/scaffold-disjoint few-shot cohort and reference pools as fixed GP',
        'model':'Neural reference-moment context network sets one GP covariance, noise and linear prior mean; warm-start scalar parameters from train-fitted Tanimoto GP',
        'labels':'Only selected/reference B labels used in conditioning; query B only in training supervision or evaluation',
        'selection':'Mean validation AP across3 independent K5 support draws; checkpoints100, patience500; no test selection',
        'acquisition':'Myopic contrast variance reduction conditional on CURRENT permitted references; ignores future change of neural hyperparameters until next reference is observed',
        'seeds':seeds,'steps':steps,'code_sha256':digest(Path(__file__))},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--models',nargs='+',default=['conditional_reg','conditional_multitask']);p.add_argument('--steps',type=int,default=2000);p.add_argument('--validation-only',action='store_true')
    a=p.parse_args();main(a.seeds,a.models,a.steps,a.validation_only)
