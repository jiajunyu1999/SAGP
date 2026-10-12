"""Validation-only training of three frozen, targeted decision-GP revisions."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit,ndtr
from controlled_joint_ablation_20261009 import batch_digest
from protected_decision_gp_20261009 import ProtectedDecisionGP,project_auxiliary,protected_classification_loss
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,query_arrays,DEVICE,SEEDS
from prepare_sar_transfer_20261007 import OUT,partition,digest
from neural_response_field_20261007 import point_cache
from train_monotone_response_20261007 import fast_point_pools,fast_point_batch
from train_joint_response_field_20261007 import contrast_cache,contrast_batch,sens_batch
from train_known_monotone_response_20261007 import RESULTS as PRETRAIN
from train_known_response_field_20261007 import metric
from response_distribution_decisions_20261007 import folded_median
from audit_response_field_20261007 import separation

RESULTS=OUT/'protected_decision_gp_20261009'
KINDS=['detached_action','aligned_action','projected_action']
BUDGETS=[1,3,5,10,20,40]


def feature_cache(build=False):
    path=RESULTS/'structure_features.npz'
    if build:
        RESULTS.mkdir(exist_ok=True)
        mapping,bits,desc=molecule_matrix(read('molecule_assay_observations.parquet'))
        np.savez_compressed(path,ids=np.array(list(mapping)),bits=bits.astype(np.uint8),desc=desc)
        print('FEATURE_CACHE',len(mapping),flush=True)
    z=np.load(path)
    return {m:i for i,m in enumerate(z['ids'])},z['bits'].astype(np.float32),z['desc']


def cohorts(seed,obs,data):
    assignment=obs.scaffold.map({s:partition(s,seed) for s in obs.scaffold.unique()})
    train=obs[assignment.eq('train')];groups={p:g for p,g in train.groupby('pair_id')}
    center=train.groupby('pair_id').signed_delta.median();known=train.groupby('pair_id').size();known=set(known[known>=30].index)
    split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
    sar={part:data.merge(split[split.partition.eq(part)][['comparison_id']],on='comparison_id') for part in ['train','validation','test']}
    frames={}
    for part in ['validation','test']:
        sens=obs[assignment.eq(part)&obs.pair_id.isin(known)].copy().reset_index(drop=True)
        sens['pair_offset']=sens.pair_id.map(center);sens['residual_abs']=abs(sens.signed_delta-sens.pair_offset);sens['binary_loss']=(sens.residual_abs>=.5).astype(int)
        frames[part]={1:sar[part],2:sens}
        for t,f in frames[part].items():separation(train,f,t,seed,part)
    return train,groups,center,known,sar,frames


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
                ids,p,gap,target=query_arrays(f,mapping);unique=np.unique(ids);lookup={m:i for i,m in enumerate(unique)}
                pm={int(m):float(v) for row,vals in zip(ids,p) for m,v in zip(row,vals)}
                pq=torch.tensor([pm[int(m)] for m in unique],device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[torch.tensor(unique,device=DEVICE)],pq)
                i=torch.tensor([lookup[m] for m in ids[:,0]],device=DEVICE);j=torch.tensor([lookup[m] for m in ids[:,1]],device=DEVICE)
                mean=torch.tensor(gap,device=DEVICE)+mu[i]-mu[j]
                var=(cov[i,i]+cov[j,j]-2*cov[i,j]).clamp_min(1e-5)
                width=float((s.max_B-s.min_B).mean())
                pred=torch.sigmoid(model.sar_risk(.3+width-mean,var)).cpu().numpy()
                key='comparison_id';o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=target;o['regression']=mean.cpu().numpy();o['variance']=var.cpu().numpy();o['prediction']=pred
                o['gaussian_probability']=ndtr((.3+width-o.regression)/np.sqrt(o.variance))
            else:
                qi=torch.tensor([mapping[m] for m in f.chemical_identity],device=DEVICE)
                pq=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[qi],pq)
                residual=mu-torch.tensor(f.pair_offset.to_numpy(np.float32),device=DEVICE);var=cov.diagonal().clamp_min(1e-5)
                q=model.magnitude(residual,var)
                key='sample_id';o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=f.residual_abs.to_numpy();o['regression']=q.cpu().numpy();o['variance']=var.cpu().numpy()
                o['prediction']=torch.sigmoid(model.sensitivity_risk(q-.5,var)).cpu().numpy()
                r=residual.cpu().numpy();sd=np.sqrt(o.variance.to_numpy());o['signed_residual_mean']=r
                o['uncalibrated_median']=folded_median(r,sd);o['gaussian_probability']=ndtr((-.5-r)/sd)+ndtr((r-.5)/sd)
            output.append(o)
    return pd.concat(output,ignore_index=True).set_index(key).loc[frame[key]].reset_index()


def evaluate(model,frames,groups,mapping,matrix):
    preds={t:infer(model,frames[t],groups,mapping,matrix,t) for t in [1,2]}
    ms={t:metric(frames[t],preds[t],t) for t in [1,2]}
    score=np.mean([ms[t][k] for t in [1,2] for k in ['AP','pair_macro_AUROC']])-.1*sum(ms[t]['macro_MAE'] for t in [1,2])
    return float(score),ms,preds


def main(args):
    RESULTS.mkdir(exist_ok=True);torch.set_num_threads(4)
    if args.build_features:feature_cache(True);return
    mapping,bits,desc=feature_cache();obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    protocol={'variants':KINDS,'seeds':SEEDS,'steps':args.steps,'GP_parameters':1483,'additional_decision_parameters':11,'GP_lr':.0003,'decision_lr':.003,'batch_size':32,
        'selection':'One variant selected globally by mean five-seed validation score; original4-ranking-minus0.1-two-MAE criterion; no candidate test scoring before selection',
        'training':'GP root NLL0.125 + SAR gap NLL0.25; sensitivity magnitude Huber0.2; binary logistic losses update11 decision parameters only',
        'variants_explanation':'detached_action blocks magnitude-to-GP gradients; aligned_action allows them; projected_action projects their negative alignment with continuous root/SAR GP density gradients',
        'projection_limit':'Raw-gradient protection, not a monotonic-loss guarantee under AdamW',
        'probabilities':'Discriminatively calibrated task probabilities; not Gaussian posterior event probabilities. GP posterior retained separately for continuous contrasts and interval uncertainty.',
        'safeguard':'Root-only warm-start and identical support/query episode streams; original five scaffold splits, references and test cohorts; no model fusion',
        'benchmark_limit':'This benchmark has already informed model design; new test evaluation is exploratory, not untouched external validation.'}
    (RESULTS/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for seed in args.seeds:
        train,groups,center,known,sar,frames=cohorts(seed,obs,data)
        point,_=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False,min_references=1)
        pp=fast_point_pools(point,BUDGETS);spool=fast_point_pools([c for c in point if c[5] in known],BUDGETS)
        cpools=contrast_cache(sar['train'],groups,mapping)
        widths={p:(g.max_B-g.min_B).to_numpy(np.float32) for p,g in groups.items()}
        # Support widths sampled with the exact same draw as contrast_batch.
        wpools={k:[(c,q,widths[c[9]]) for c,q in cpools[k]] for k in BUDGETS}
        cp_path=PRETRAIN/f'conditional_monotone_seed_{seed}.pt';cp=torch.load(cp_path,map_location=DEVICE,weights_only=False)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-cp['desc_mean'])/cp['desc_std'],-8,8)],1),device=DEVICE)
        for kind in args.models:
            done=RESULTS/f'complete_{kind}_seed_{seed}.json'
            if done.exists():print('SKIP',seed,kind,flush=True);continue
            start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=ProtectedDecisionGP(cp['dim']).to(DEVICE);missing,unexpected=model.load_state_dict(cp['state_dict'],strict=False)
            assert not unexpected and len(missing)==11
            gp=model.gp_parameters();heads=model.decision_parameters()
            assert sum(p.numel() for p in gp)==1483 and sum(p.numel() for p in heads)==11
            opt=torch.optim.AdamW([{'params':gp,'lr':.0003,'weight_decay':.003},{'params':heads,'lr':.003,'weight_decay':0}])
            best,ms,_=evaluate(model,frames['validation'],groups,mapping,matrix);state=copy.deepcopy(model.state_dict());best_step=0
            hist=[{'step':0,'score':best,**{f'task{t}_{k}':ms[t][k] for t in [1,2] for k in ['AP','pair_macro_AUROC','macro_MAE']}}]
            bh=hashlib.sha256();conflicts=0
            for step in range(args.steps):
                model.train();k=int(rng.choice(BUDGETS));before=copy.deepcopy(rng.bit_generator.state)
                cb,labels=contrast_batch(cpools,matrix,rng,k);after=copy.deepcopy(rng.bit_generator.state)
                rng.bit_generator.state=before;w=[]
                for _ in range(32):
                    c,q,width=wpools[k][rng.integers(len(wpools[k]))];i=int(rng.choice(q));support=rng.choice(c[7][i],k,replace=False);w.append(float(width[support].mean()))
                assert rng.bit_generator.state==after
                sb,offset=sens_batch(spool,matrix,center.to_dict(),rng,k);rb=fast_point_batch(pp,matrix,rng,k)
                batch_digest(bh,cb,(labels,),sb,(offset,),rb)
                width=torch.tensor(w,device=DEVICE,dtype=torch.float32)
                mean,var=model.contrast(*cb[:-1]);mu,sv=model.point(*sb[:-1]);pm,pv=model.point(*rb[:-1])
                density=.25*(var.log()+(cb[-1]-mean).square()/var).mean()+.125*(pv.log()+(rb[-1]-pm).square()/pv).mean()
                residual=mu-offset;truth=abs(sb[-1]-offset)
                q=model.magnitude(residual.detach() if kind=='detached_action' else residual,sv.detach() if kind=='detached_action' else sv)
                magnitude=.2*F.smooth_l1_loss(q,truth)
                classification=protected_classification_loss(model,mean,var,width,labels,q,sv,(truth>=.5).float())
                loss=density+magnitude+classification
                opt.zero_grad(set_to_none=True)
                if kind=='projected_action':
                    gb=torch.autograd.grad(density,gp,retain_graph=True,allow_unused=True)
                    ga=torch.autograd.grad(magnitude,gp,retain_graph=True,allow_unused=True)
                    gb=[torch.zeros_like(p) if g is None else g for p,g in zip(gp,gb)]
                    ga=[torch.zeros_like(p) if g is None else g for p,g in zip(gp,ga)]
                    projected,dot=project_auxiliary(gb,ga);conflicts+=int(dot<0)
                    loss.backward()
                    for p,b,a in zip(gp,gb,projected):p.grad=b+a
                else:loss.backward()
                assert torch.isfinite(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0:
                    val,ms,_=evaluate(model,frames['validation'],groups,mapping,matrix)
                    hist.append({'step':step+1,'score':val,'loss':float(loss.detach()),'density_loss':float(density.detach()),'magnitude_loss':float(magnitude.detach()),'classification_loss':float(classification.detach()),'conflicts':conflicts,**{f'task{t}_{key}':ms[t][key] for t in [1,2] for key in ['AP','pair_macro_AUROC','macro_MAE']}})
                    if val>best+1e-5:best=val;state=copy.deepcopy(model.state_dict());best_step=step+1
                    print(seed,kind,step+1,'VALIDATION',round(val,5),flush=True)
            final=copy.deepcopy(model.state_dict());rows=[]
            for ck,s,selected_step in [('selected',state,best_step),('final',final,args.steps)]:
                model.load_state_dict(s);model.eval();_,ms,preds=evaluate(model,frames['validation'],groups,mapping,matrix)
                torch.save({**cp,'state_dict':s,'variant':kind,'best_step':selected_step,'steps':args.steps},RESULTS/f'{kind}_{ck}_seed_{seed}.pt')
                for t in [1,2]:
                    preds[t].to_csv(RESULTS/f'task{t}_{kind}_{ck}_validation_seed_{seed}.csv',index=False)
                    rows.append({'seed':seed,'model':kind,'checkpoint':ck,'partition':'validation','task':t,'best_step':selected_step,**ms[t]})
            pd.DataFrame(rows).to_csv(RESULTS/f'validation_metrics_{kind}_seed_{seed}.csv',index=False);pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            record={'seed':seed,'model':kind,'steps':args.steps,'best_step':best_step,'validation_score':best,'batch_sha256':bh.hexdigest(),'parameters':1494,'conflicts':conflicts,'seconds':time.monotonic()-start,'pretrain_sha256':digest(cp_path),'code_sha256':digest(Path(__file__))}
            done.write_text(json.dumps(record,indent=2));print('COMPLETE',json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--models',nargs='+',choices=KINDS,default=KINDS);p.add_argument('--steps',type=int,default=1000);p.add_argument('--build-features',action='store_true')
    main(p.parse_args())
