"""Validation-only training of three frozen, targeted decision-GP revisions."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import expit,ndtr
from controlled_joint_ablation_20261009 import batch_digest
from contextual_decision_gp_20261009 import ContextualDecisionGP,reference_context
from train_protected_decision_gp_20261009 import feature_cache
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,query_arrays,DEVICE,SEEDS
from prepare_sar_transfer_20261007 import OUT,partition,digest
from neural_response_field_20261007 import point_cache
from train_monotone_response_20261007 import fast_point_pools,fast_point_batch
from train_joint_response_field_20261007 import contrast_cache,contrast_batch,sens_batch
from train_known_monotone_response_20261007 import RESULTS as PRETRAIN
from train_known_response_field_20261007 import metric
from response_distribution_decisions_20261007 import folded_median
from audit_response_field_20261007 import separation

RESULTS=OUT/'contextual_decision_gp_20261009'
KINDS=['context_linear','context_neural']
BUDGETS=[1,3,5,10,20,40]



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
                features=reference_context(matrix[si],ps,ys,matrix[torch.tensor(unique,device=DEVICE)],pq,ys.median(),cov.diagonal())
                cf=(features[i]+features[j])/2
                pred=torch.sigmoid(model.sar_logits(mean,var,width,cf)).cpu().numpy()
                key='comparison_id';o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=target;o['regression']=mean.cpu().numpy();o['variance']=var.cpu().numpy();o['prediction']=pred
                o['gaussian_probability']=ndtr((.3+width-o.regression)/np.sqrt(o.variance))
            else:
                qi=torch.tensor([mapping[m] for m in f.chemical_identity],device=DEVICE)
                pq=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[qi],pq)
                residual=mu-torch.tensor(f.pair_offset.to_numpy(np.float32),device=DEVICE);var=cov.diagonal().clamp_min(1e-5)
                features=reference_context(matrix[si],ps,ys,matrix[qi],pq,torch.as_tensor(float(f.pair_offset.iloc[0]),device=DEVICE),var)
                q=model.sensitivity_action(residual,var,features)
                key='sample_id';o=f[[key,'pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=f.residual_abs.to_numpy();o['regression']=q.cpu().numpy();o['variance']=var.cpu().numpy()
                o['prediction']=torch.sigmoid(model.sensitivity_logits(q,var,features)).cpu().numpy()
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
    protocol={'variants':KINDS,'seeds':SEEDS,'steps':args.steps,'GP_parameters':1483,
        'selection':'one global validation-selected variant, original common criterion, before candidate test evaluation',
        'features':'reference mean/std absolute centered shift; signed shift std; source potency position; max Tanimoto; similarity-weighted shift std; similarity-weighted source-potency distance; log variance',
        'training':'same density losses and episode tensors as protected GP; continuous/contextual action and probability adapters use detached GP moments; independent gradient clipping for GP and decision parameters',
        'benchmark_limit':'exploratory re-evaluation of an already explored benchmark; no prediction ensemble'}
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
            model=ContextualDecisionGP(cp['dim'],kind).to(DEVICE);missing,unexpected=model.load_state_dict(cp['state_dict'],strict=False)
            assert not unexpected
            gp=model.gp_parameters();heads=model.decision_parameters()
            assert sum(p.numel() for p in gp)==1483
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
                cm,cc=model.posterior(*cb[:5]);mean=cb[5]+cm[...,0]-cm[...,1];var=(cc[...,0,0]+cc[...,1,1]-2*cc[...,0,1]).clamp_min(1e-5)
                mu,sv=model.point(*sb[:-1]);pm,pv=model.point(*rb[:-1])
                density=.25*(var.log()+(cb[-1]-mean).square()/var).mean()+.125*(pv.log()+(rb[-1]-pm).square()/pv).mean()
                residual=mu-offset;truth=abs(sb[-1]-offset)
                cf=reference_context(cb[0],cb[1],cb[2],cb[3],cb[4],cb[2].median(-1).values,cc.diagonal(dim1=-2,dim2=-1)).mean(-2)
                sf=reference_context(sb[0],sb[1],sb[2],sb[3],sb[4],offset,sv.unsqueeze(-1)).squeeze(-2)
                q=model.sensitivity_action(residual,sv,sf)
                magnitude=.2*F.smooth_l1_loss(q,truth)
                classification=F.binary_cross_entropy_with_logits(model.sar_logits(mean,var,width,cf),labels)+F.binary_cross_entropy_with_logits(model.sensitivity_logits(q,sv,sf),(truth>=.5).float())
                loss=density+magnitude+classification
                opt.zero_grad(set_to_none=True);loss.backward()
                assert torch.isfinite(loss)
                torch.nn.utils.clip_grad_norm_(gp,5);torch.nn.utils.clip_grad_norm_(heads,5);opt.step()
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
