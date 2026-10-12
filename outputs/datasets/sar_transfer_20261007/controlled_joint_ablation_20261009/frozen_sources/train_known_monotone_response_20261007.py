"""Reference-rich monotone response GP on original known-pair tasks1/2."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from meta_gp_sar_20261007 import read,molecule_matrix,context_components,query_arrays,DEVICE,SEEDS
from prepare_sar_transfer_20261007 import OUT,partition,digest
from neural_response_field_20261007 import point_cache
from monotone_response_gp_20261007 import MonotoneResponseGP
from train_monotone_response_20261007 import fast_point_pools,fast_point_batch
from train_known_response_field_20261007 import metric
from audit_response_field_20261007 import separation
RESULTS=OUT/'models_known_monotone_response_20261007';RESULTS.mkdir(exist_ok=True)


def infer(model,frame,groups,mapping,matrix,task,cap=100):
    output=[];model.eval()
    with torch.no_grad():
        for pair,f in frame.groupby('pair_id',sort=False):
            support=groups[pair]
            if len(support)>cap:support=support.sample(cap,random_state=17)
            si=torch.tensor([mapping[m] for m in support.chemical_identity],device=DEVICE)
            ps=torch.tensor(((support.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
            ys=torch.tensor((support.p_B-support.p_A).to_numpy(np.float32),device=DEVICE)
            if task==1:
                ids,p,gap,target=query_arrays(f,mapping)
                unique=np.unique(ids);lookup={m:i for i,m in enumerate(unique)}
                pm={int(m):float(v) for row,vals in zip(ids,p) for m,v in zip(row,vals)}
                qs=np.array([lookup[m] for m in ids[:,0]]);qw=np.array([lookup[m] for m in ids[:,1]])
                pq=torch.tensor([pm[int(m)] for m in unique],device=DEVICE)
                mu,cov=model.posterior(matrix[si],ps,ys,matrix[torch.tensor(unique,device=DEVICE)],pq)
                mu=mu.cpu().numpy();cov=cov.cpu().numpy()
                mean=gap+mu[qs]-mu[qw];var=(cov[qs,qs]+cov[qw,qw]-2*cov[qs,qw]).clip(1e-5)
                width=float((support.max_B-support.min_B).mean());pred=ndtr((.3-mean+width)/np.sqrt(var))
                o=f[['comparison_id','pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=target;o['regression']=mean;o['prediction']=pred;o['variance']=var
            else:
                qi=torch.tensor([mapping[m] for m in f.chemical_identity],device=DEVICE)
                pq=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
                mean,cov=model.posterior(matrix[si],ps,ys,matrix[qi],pq)
                mu=mean.cpu().numpy()-f.pair_offset.to_numpy();sd=np.sqrt(np.diag(cov.cpu().numpy()).clip(1e-5))
                pred=ndtr((-.5-mu)/sd)+ndtr((mu-.5)/sd)
                absolute=sd*np.sqrt(2/np.pi)*np.exp(-.5*(mu/sd)**2)+mu*(2*ndtr(mu/sd)-1)
                o=f[['sample_id','pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=f.residual_abs.to_numpy();o['regression']=absolute;o['prediction']=pred;o['variance']=sd**2
            output.append(o)
    key='comparison_id' if task==1 else 'sample_id'
    return pd.concat(output,ignore_index=True).set_index(key).loc[frame[key]].reset_index()


def val_shift(model,frame,groups,mapping,matrix,cap=100):
    vals=[]
    with torch.no_grad():
        for pair,f in frame.groupby('pair_id'):
            if pair not in groups or len(groups[pair])<3:continue
            support=groups[pair]
            if len(support)>cap:support=support.sample(cap,random_state=17)
            if len(f)>32:f=f.sample(32,random_state=17)
            si=torch.tensor([mapping[m] for m in support.chemical_identity],device=DEVICE)
            ps=torch.tensor(((support.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
            ys=torch.tensor((support.p_B-support.p_A).to_numpy(np.float32),device=DEVICE)
            qi=torch.tensor([mapping[m] for m in f.chemical_identity],device=DEVICE)
            pq=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
            mu,cov=model.posterior(matrix[si],ps,ys,matrix[qi],pq);v=cov.diagonal().clamp_min(1e-5)
            target=torch.tensor((f.p_B-f.p_A).to_numpy(np.float32),device=DEVICE)
            vals.append(float(.5*(v.log()+(target-mu).square()/v).mean()))
    return float(np.mean(vals))


def main(seeds,kinds,steps,test):
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=[];path=RESULTS/'metrics.csv'
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.model.isin(kinds))].to_dict('records')
    for seed in seeds:
        cached,train=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False)
        pools=fast_point_pools(cached,(5,10,20,40));groups={p:g for p,g in train.groupby('pair_id')}
        sp=obs.scaffold.map(lambda s:partition(s,seed));val=obs[sp.eq('validation')]
        trainids=[mapping[m] for m in train.chemical_identity.unique()];dm=desc[trainids].mean(0);ds=desc[trainids].std(0).clip(.01)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-dm)/ds,-8,8)],1),device=DEVICE)
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        sar={p:data.merge(split[split.partition.eq(p)][['comparison_id']],on='comparison_id') for p in ['validation','test']}
        known=train.groupby('pair_id').size();known=set(known[known>=30].index);center=train.groupby('pair_id').signed_delta.median()
        sens={p:obs[sp.eq(p)&obs.pair_id.isin(known)].copy().reset_index(drop=True) for p in ['validation','test']}
        for f in sens.values():f['pair_offset']=f.pair_id.map(center);f['residual_abs']=abs(f.signed_delta-f.pair_offset);f['binary_loss']=(f.residual_abs>=.5).astype(int)
        for p in ['validation','test']:
            separation(train,sar[p],1,seed,p);separation(train,sens[p],2,seed,p)
        for kind in kinds:
            start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            if kind=='heteroscedastic_monotone':
                from heteroscedastic_response_gp_20261007 import HeteroscedasticResponseGP
                model=HeteroscedasticResponseGP(matrix.shape[1]).to(DEVICE)
            else:model=MonotoneResponseGP(matrix.shape[1],kind=='conditional_monotone').to(DEVICE)
            opt=torch.optim.AdamW([
                {'params':[p for n,p in model.named_parameters() if n.startswith(('context.','mean_warp.','variance_network.')) and p.requires_grad],'lr':.0007,'weight_decay':.003},
                {'params':[p for n,p in model.named_parameters() if not n.startswith(('context.','mean_warp.','variance_network.')) and p.requires_grad],'lr':.007,'weight_decay':0.}])
            best=float('inf');best_step=0;state=None;hist=[]
            for step in range(steps):
                model.train();batch=fast_point_batch(pools,matrix,rng,int(rng.choice([5,10,20,40])),batch=64)
                mu,v=model.point(*batch[:-1]);loss=.5*(v.log()+(batch[-1]-mu).square()/v).mean()
                loss=loss+.005*model.warp_parameters(batch[1],batch[2])[2].log().square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0:
                    model.eval();score=val_shift(model,val,groups,mapping,matrix)
                    hist.append({'step':step+1,'loss':float(loss.detach()),'validation_macro_shift_NLL':score})
                    print(seed,kind,step+1,'val NLL',round(score,4),flush=True)
                    if score<best-1e-5:best=score;best_step=step+1;state=copy.deepcopy(model.state_dict())
                    if step+1-best_step>=400:break
            model.load_state_dict(state);model.eval()
            torch.save({'state_dict':model.state_dict(),'dim':matrix.shape[1],'desc_mean':dm,'desc_std':ds,'best_step':best_step},RESULTS/f'{kind}_seed_{seed}.pt')
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            for task,frames in [(1,sar),(2,sens)]:
                for part in (['validation','test'] if test else ['validation']):
                    pred=infer(model,frames[part],groups,mapping,matrix,task)
                    pred.to_csv(RESULTS/f'task{task}_{kind}_{part}_seed_{seed}.csv',index=False)
                    rows.append({'seed':seed,'task':task,'model':kind,'partition':part,'best_step':best_step,**metric(frames[part],pred,task)})
            pd.DataFrame(rows).to_csv(path,index=False)
            print(seed,kind,'finished',round(time.monotonic()-start,1),'seconds',flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps({'model':'Single GP,9-slope monotone potency-axis mean,reference-conditioned neural warp/covariance vsglobal warp control',
        'training':'Root signed-shift episodes,K5/10/20/40,64queries;exclude query scaffold from all references',
        'selection':'Validation pair-macro shift NLL with deployment-matched <=100 refs',
        'evaluation':'Original known-pair task1/2 cohorts;one shared field predicts both;no few-shot warm-start',
        'seeds':seeds,'models':kinds,'max_steps':steps,'test_evaluated':test,'code_sha256':digest(Path(__file__))},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--models',nargs='+',default=['global_monotone','conditional_monotone']);p.add_argument('--steps',type=int,default=1800);p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args();main(a.seeds,a.models,a.steps,a.evaluate_test)
