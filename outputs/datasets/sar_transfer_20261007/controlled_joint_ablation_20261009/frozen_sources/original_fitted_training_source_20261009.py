"""Frozen, matched-episode Joint vs single-task ablation; no test selection."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr
from meta_gp_sar_20261007 import read, molecule_matrix, context_components, query_arrays, DEVICE, SEEDS
from prepare_sar_transfer_20261007 import OUT, partition, digest
from neural_response_field_20261007 import point_cache, moments
from monotone_response_gp_20261007 import MonotoneResponseGP
from train_monotone_response_20261007 import fast_point_pools, fast_point_batch
from train_joint_response_field_20261007 import contrast_cache, contrast_batch, sens_batch, probit_event_nll, folded_event_nll, gaussian_folded_median
from train_known_monotone_response_20261007 import RESULTS as PRETRAIN
from train_known_response_field_20261007 import metric
from audit_response_field_20261007 import separation
from response_distribution_decisions_20261007 import folded_median

RESULTS=OUT/'controlled_joint_ablation_20261009'
MODELS=['task1_only','task2_only','joint','no_molecular_residual','no_monotone_mean','no_contrast_covariance','shared_multihead']
BUDGETS=[1,3,5,10,20,40]


class ControlledGP(MonotoneResponseGP):
    def __init__(self,dim,kind):
        super().__init__(dim)
        self.kind=kind
        if kind=='shared_multihead':
            # Same GP encoder and references; task-specific moment readouts need
            # not correspond to one shared signed-shift posterior. +62 parameters.
            self.head_trunk=nn.Sequential(nn.Linear(6,4),nn.SiLU())
            self.heads=nn.ModuleList([nn.Linear(4,2) for _ in range(3)])
            for head in self.heads:
                nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def kernel(self,x,p,y,q,state=None):
        if self.kind!='no_molecular_residual':return super().kernel(x,p,y,q,state)
        offset=self.log_offset.clamp(-3,3).exp().square()
        slope=self.log_slope.clamp(-4,2).exp().square()
        if state is not None:slope=slope*state[...,0].exp()
        return offset+slope[...,None,None]*p.unsqueeze(-1)*q.unsqueeze(-2)

    def warp_parameters(self,ps,ys):
        if self.kind!='no_monotone_mean':return super().warp_parameters(ps,ys)
        pm=ps.mean(-1,keepdim=True)
        scale=(ps-pm).square().mean(-1,keepdim=True).sqrt().clamp_min(.25)
        strength=min(1.,max(0.,(ps.shape[-1]-1)/2))
        # Same nine-slope network and smooth hinges; signed slopes permit
        # nonmonotone response. Bounds protect extrapolation numerically.
        slopes=(self.log_slopes+strength*2*torch.tanh(self.mean_warp(moments(ps,ys)))).clamp(-2,1.5)+1
        return pm,scale,slopes

    def point_moments(self,mu,var,which=0):
        var=var.clamp_min(1e-5)
        if self.kind!='shared_multihead':return mu,var
        zeros=torch.zeros_like(mu)
        z=torch.stack([mu,zeros,var.log(),zeros,zeros,zeros],-1).clamp(-12,12)
        delta=self.heads[which](self.head_trunk(z))
        return mu+delta[...,0],var*delta[...,1].clamp(-3,3).exp()

    def contrast_moments(self,mu1,mu2,v1,v2,cross,gap):
        mean=gap+mu1-mu2
        var=(v1+v2-(0 if self.kind=='no_contrast_covariance' else 2*cross)).clamp_min(1e-5)
        if self.kind=='shared_multihead':
            z=torch.stack([mu1,mu2,v1.clamp_min(1e-5).log(),v2.clamp_min(1e-5).log(),cross,gap],-1).clamp(-12,12)
            delta=self.heads[1](self.head_trunk(z))
            mean=mean+delta[...,0];var=var*delta[...,1].clamp(-3,3).exp()
        return mean,var

    def contrast(self,xs,ps,ys,xq,pq,gap):
        mu,cov=self.posterior(xs,ps,ys,xq,pq)
        return self.contrast_moments(mu[...,0],mu[...,1],cov[...,0,0],cov[...,1,1],cov[...,0,1],gap)

    def point(self,xs,ps,ys,xq,pq):
        mu,cov=self.posterior(xs,ps,ys,xq,pq)
        return self.point_moments(mu[...,0],cov[...,0,0],0)

    def sensitivity(self,xs,ps,ys,xq,pq):
        mu,cov=self.posterior(xs,ps,ys,xq,pq)
        return self.point_moments(mu[...,0],cov[...,0,0],2)


def infer(model,frame,groups,mapping,matrix,task):
    output=[];model.eval()
    with torch.no_grad():
        for pair,f in frame.groupby('pair_id',sort=False):
            s=groups[pair]
            if len(s)>100:s=s.sample(100,random_state=17)
            si=torch.tensor([mapping[m] for m in s.chemical_identity],device=DEVICE)
            ps=torch.tensor(((s.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
            ys=torch.tensor(s.signed_delta.to_numpy(np.float32),device=DEVICE)
            if task==1:
                ids,p,gap,target=query_arrays(f,mapping)
                unique=np.unique(ids);lookup={m:i for i,m in enumerate(unique)}
                pm={int(m):float(v) for row,vals in zip(ids,p) for m,v in zip(row,vals)}
                pq=torch.tensor([pm[int(m)] for m in unique],device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[torch.tensor(unique,device=DEVICE)],pq)
                i=torch.tensor([lookup[m] for m in ids[:,0]],device=DEVICE)
                j=torch.tensor([lookup[m] for m in ids[:,1]],device=DEVICE)
                mean,var=model.contrast_moments(mu[i],mu[j],cov[i,i],cov[j,j],cov[i,j],torch.tensor(gap,device=DEVICE))
                mean=mean.cpu().numpy();var=var.cpu().numpy()
                width=float((s.max_B-s.min_B).mean())
                pred=ndtr((.3-mean+width)/np.sqrt(var));key='comparison_id'
                o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=target;o['regression']=mean;o['prediction']=pred;o['variance']=var
            else:
                qi=torch.tensor([mapping[m] for m in f.chemical_identity],device=DEVICE)
                pq=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[qi],pq)
                mu,var=model.point_moments(mu,cov.diagonal(),2)
                residual=mu.cpu().numpy()-f.pair_offset.to_numpy();sd=np.sqrt(var.cpu().numpy())
                key='sample_id';o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=f.residual_abs.to_numpy();o['regression']=folded_median(residual,sd)
                o['prediction']=ndtr((-.5-residual)/sd)+ndtr((residual-.5)/sd)
                o['variance']=sd**2;o['signed_residual_mean']=residual
            output.append(o)
    return pd.concat(output,ignore_index=True).set_index(key).loc[frame[key]].reset_index()


def evaluate(model,frames,groups,mapping,matrix):
    preds={t:infer(model,frames[t],groups,mapping,matrix,t) for t in [1,2]}
    metrics={t:metric(frames[t],preds[t],t) for t in [1,2]}
    value=np.mean([metrics[t][k] for t in [1,2] for k in ['AP','pair_macro_AUROC']])-.1*sum(metrics[t]['macro_MAE'] for t in [1,2])
    return float(value),metrics,preds


def batch_digest(h,*batches):
    for batch in batches:
        for value in batch:
            if isinstance(value,torch.Tensor):h.update(value.detach().cpu().numpy().tobytes())


def main(args):
    RESULTS.mkdir(exist_ok=True);torch.set_num_threads(4)
    obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs)
    plan={'seeds':SEEDS,'models':MODELS,'steps':args.steps,'batch_size_per_loss':32,'training_budgets':BUDGETS,'inference_reference_cap':100,
          'optimizer':'AdamW lr=0.0003 weight_decay=0.003 clip=5','checkpoints':'0 and every100 steps; no early stopping',
          'selection':'same mean of validation Task1/2 AP and pairAUROC minus0.1 sum macroMAEs for all models; also report final-budget checkpoints',
          'warm_start':'same root-shift-only conditional_monotone checkpoint per seed; structural ablations change the forward function',
          'single_task_interpretation':'A/B vs C estimates benefit from additional task supervision, not equal-label information',
          'multihead_interpretation':'shared GP/neural reference encoder with independent task moment heads (+58 parameters); matched labels, root pool, references and steps; not a GP-free neural comparator',
          'code_sha256':digest(Path(__file__))}
    (RESULTS/'protocol.json').write_text(json.dumps(plan,indent=2))
    for seed in args.seeds:
        point,train=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False,min_references=1)
        groups={p:g for p,g in train.groupby('pair_id')};center=train.groupby('pair_id').signed_delta.median()
        known=train.groupby('pair_id').size();known=set(known[known>=30].index)
        pp=fast_point_pools(point,BUDGETS);spool=fast_point_pools([c for c in point if c[5] in known],BUDGETS)
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        sar={p:data.merge(split[split.partition.eq(p)][['comparison_id']],on='comparison_id') for p in ['train','validation','test']}
        cpools=contrast_cache(sar['train'],groups,mapping)
        assignment=obs.scaffold.map({s:partition(s,seed) for s in obs.scaffold.unique()})
        frames={}
        for part in ['validation','test']:
            sens=obs[assignment.eq(part)&obs.pair_id.isin(known)].copy().reset_index(drop=True)
            sens['pair_offset']=sens.pair_id.map(center);sens['residual_abs']=abs(sens.signed_delta-sens.pair_offset);sens['binary_loss']=(sens.residual_abs>=.5).astype(int)
            frames[part]={1:sar[part],2:sens}
            for t,f in frames[part].items():separation(train,f,t,seed,part)
        refs=[]
        for pair,s in groups.items():
            if len(s)>100:s=s.sample(100,random_state=17)
            refs.extend({'seed':seed,'pair_id':pair,'order':i,'sample_id':r.sample_id,'chemical_identity':r.chemical_identity} for i,r in enumerate(s.itertuples()))
        pd.DataFrame(refs).to_csv(RESULTS/f'reference_manifest_seed_{seed}.csv',index=False)
        cp_path=PRETRAIN/f'conditional_monotone_seed_{seed}.pt'
        cp=torch.load(cp_path,map_location=DEVICE,weights_only=False)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
        audit={'seed':seed,'pretrain_sha256':digest(cp_path),'root_pool_hash':digest(OUT/'molecule_assay_observations.parquet'),'split_sha256':digest(OUT/f'anchor_split_scaffold_seed_{seed}.parquet'),
               'train_rows':len(train),'test_counts':{str(t):len(f) for t,f in frames['test'].items()},'models':{}}
        for kind in args.models:
            done=RESULTS/f'complete_{kind}_seed_{seed}.json'
            if done.exists() and not args.overwrite:
                print('RESUME skip',seed,kind,flush=True);continue
            started=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=ControlledGP(cp['dim'],kind).to(DEVICE)
            missing,unexpected=model.load_state_dict(cp['state_dict'],strict=False)
            assert not unexpected and all(n.startswith(('head_trunk','heads')) for n in missing)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.003)
            best,vm,_=evaluate(model,frames['validation'],groups,mapping,matrix)
            best_state=copy.deepcopy(model.state_dict());best_step=0
            history=[{'step':0,'score':best,**{f'task{t}_{k}':vm[t][k] for t in [1,2] for k in ['AP','pair_macro_AUROC','macro_MAE']}}]
            bh=hashlib.sha256()
            for step in range(args.steps):
                model.train();k=int(rng.choice(BUDGETS))
                cb,labels=contrast_batch(cpools,matrix,rng,k)
                sb,offset=sens_batch(spool,matrix,center.to_dict(),rng,k)
                rb=fast_point_batch(pp,matrix,rng,k)
                # Draw every stream even when its loss is disabled. All seven
                # variants receive byte-identical query/support tensors.
                batch_digest(bh,cb,(labels,),sb,(offset,),rb)
                pm,pv=model.point(*rb[:-1]);loss=.125*(pv.log()+(rb[-1]-pm).square()/pv).mean()
                l1=l2=torch.tensor(0.,device=DEVICE)
                if kind!='task2_only':
                    mean,v=model.contrast(*cb[:-1])
                    l1=.25*(v.log()+(cb[-1]-mean).square()/v).mean()+probit_event_nll((.3-mean)/v.sqrt(),labels)
                    loss=loss+l1
                if kind!='task1_only':
                    mu,var=model.sensitivity(*sb[:-1]);sd=var.sqrt();res=mu-offset;truth=abs(sb[-1]-offset)
                    l2=folded_event_nll(res,sd,(truth>=.5).float())+.2*F.smooth_l1_loss(gaussian_folded_median(res,sd),truth)
                    loss=loss+l2
                assert torch.isfinite(loss), (seed,kind,step)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0:
                    val,vm,_=evaluate(model,frames['validation'],groups,mapping,matrix)
                    history.append({'step':step+1,'score':val,'loss':float(loss.detach()),'task1_loss':float(l1.detach()),'task2_loss':float(l2.detach()),**{f'task{t}_{key}':vm[t][key] for t in [1,2] for key in ['AP','pair_macro_AUROC','macro_MAE']}})
                    if val>best+1e-5:best=val;best_state=copy.deepcopy(model.state_dict());best_step=step+1
                    print(seed,kind,step+1,'validation',round(val,5),flush=True)
            rows=[];final_state=copy.deepcopy(model.state_dict())
            for checkpoint,state,selected_step in [('selected',best_state,best_step),('final',final_state,args.steps)]:
                model.load_state_dict(state);model.eval()
                torch.save({**cp,'state_dict':state,'model':kind,'best_step':selected_step,'max_steps':args.steps},RESULTS/f'{kind}_{checkpoint}_seed_{seed}.pt')
                for part in ['validation','test']:
                    _,ms,preds=evaluate(model,frames[part],groups,mapping,matrix)
                    for t in [1,2]:
                        # Exact benchmark test cohort and labels, not a subset.
                        original=OUT/'models_joint_response_field_20261007'/f'task{t}_joint_monotone_{part}_seed_{seed}.csv'
                        old=pd.read_csv(original);key='comparison_id' if t==1 else 'sample_id'
                        aligned=old.set_index(key).loc[preds[t][key]]
                        assert len(old)==len(preds[t]) and np.array_equal(aligned.binary_loss.to_numpy(),preds[t].binary_loss.to_numpy())
                        preds[t].to_csv(RESULTS/f'task{t}_{kind}_{checkpoint}_{part}_seed_{seed}.csv',index=False)
                        rows.append({'seed':seed,'model':kind,'checkpoint':checkpoint,'partition':part,'task':t,'best_step':selected_step,**ms[t]})
            pd.DataFrame(rows).to_csv(RESULTS/f'metrics_{kind}_seed_{seed}.csv',index=False)
            pd.DataFrame(history).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            record={'seed':seed,'model':kind,'steps':args.steps,'best_step':best_step,'parameters':sum(p.numel() for p in model.parameters()),'batch_sha256':bh.hexdigest(),'seconds':time.monotonic()-started,'code_sha256':digest(Path(__file__))}
            done.write_text(json.dumps(record,indent=2));audit['models'][kind]=record
            (RESULTS/f'control_audit_seed_{seed}.json').write_text(json.dumps(audit,indent=2))
            print('COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);p.add_argument('--steps',type=int,default=1000);p.add_argument('--overwrite',action='store_true')
    main(p.parse_args())
