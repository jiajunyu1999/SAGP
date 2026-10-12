"""One signed-response neural GP evaluated on known-pair tasks 1 and 2.

Train on molecule-level IC50 shift episodes mined from all paired observations.
Global scaffold assignments match the existing task1/task2 benchmarks. One
shared signed field per split provides both SAR and molecular-sensitivity
outputs; no combination of independently fitted predictors.
"""
import argparse,copy,json,time
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.metrics import average_precision_score
from prepare_sar_transfer_20261007 import OUT,SEEDS,partition,digest
from meta_gp_sar_20261007 import read,context_components,molecule_matrix,query_arrays,DEVICE
from conditional_neural_gp_20261007 import ConditionalGP
from neural_response_field_20261007 import NeuralResponseFieldGP,point_cache,point_batch,conditional_point
from train_anchored_sar_transfer_20261007 import metrics as sar_metrics
from deep_three_tasks_20261007 import sensitivity_metrics
from pathlib import Path
RESULTS=OUT/'models_known_response_field_20261007';RESULTS.mkdir(exist_ok=True)


def inference(model,frame,trainobs,mapping,matrix,task,max_references=100):
    outputs=[];model.eval();groups={p:g for p,g in trainobs.groupby('pair_id')}
    with torch.no_grad():
        for pair,f in frame.groupby('pair_id',sort=False):
            support=groups[pair]
            if len(support)>max_references:support=support.sample(max_references,random_state=17)
            si=torch.tensor([mapping[m] for m in support.chemical_identity],device=DEVICE)
            ps=torch.tensor(((support.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)
            ys=torch.tensor((support.p_B-support.p_A).to_numpy(np.float32),device=DEVICE)
            if task==1:
                ids,p,gap,target=query_arrays(f,mapping)
                n=len(f);xs=matrix[si][None].expand(n,-1,-1);pp=ps[None].expand(n,-1);yy=ys[None].expand(n,-1)
                mean,var=model.contrast(xs,pp,yy,matrix[torch.tensor(ids,device=DEVICE)],torch.tensor(p,device=DEVICE),torch.tensor(gap,device=DEVICE))
                mean=mean.cpu().numpy();var=var.cpu().numpy()
                width=float((support.max_B-support.min_B).mean())
                pred=ndtr((.3-mean+width)/np.sqrt(var));o=f[['comparison_id','pair_id','target_chembl_id','binary_loss']].copy()
                o['target']=target;o['regression']=mean;o['variance']=var;o['prediction']=pred
            else:
                ids=torch.tensor([[mapping[m]] for m in f.chemical_identity],device=DEVICE)
                p=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32)[:,None],device=DEVICE)
                n=len(f);xs=matrix[si][None].expand(n,-1,-1);pp=ps[None].expand(n,-1);yy=ys[None].expand(n,-1)
                mu,var=conditional_point(model,xs,pp,yy,matrix[ids],p)
                mu=mu.cpu().numpy()-f.pair_offset.to_numpy();sd=np.sqrt(var.cpu().numpy())
                pred=ndtr((-.5-mu)/sd)+ndtr((mu-.5)/sd)
                absolute=sd*np.sqrt(2/np.pi)*np.exp(-.5*(mu/sd)**2)+mu*(2*ndtr(mu/sd)-1)
                o=f[['sample_id','pair_id','target_chembl_id','binary_loss']].copy();o['prediction']=pred;o['target']=f.residual_abs.to_numpy();o['regression']=absolute;o['variance']=sd**2
            outputs.append(o)
    return pd.concat(outputs,ignore_index=True).set_index('comparison_id' if task==1 else 'sample_id').loc[frame['comparison_id' if task==1 else 'sample_id']].reset_index()


def metric(frame,out,task):
    if task==2:return sensitivity_metrics(frame,out.prediction.to_numpy(),out.regression.to_numpy())
    m=sar_metrics(frame,out.prediction.to_numpy());e=abs(out.target-out.regression)
    m['MAE']=float(e.mean());m['macro_MAE']=float(pd.DataFrame({'pair_id':out.pair_id,'e':e}).groupby('pair_id').e.mean().mean());return m


def validation_point(model,val,groups,mapping,matrix):
    # A fixed label-blind reference subset per context; query outcomes only
    # score point predictions and are never used for support selection.
    vals=[]
    with torch.no_grad():
        for pair,f in val.groupby('pair_id'):
            if pair not in groups:continue
            support=groups[pair]
            if len(support)<3:continue
            support=support.sample(min(10,len(support)),random_state=17)
            if len(f)>32:f=f.sample(32,random_state=17)
            ids=torch.tensor([mapping[m] for m in support.chemical_identity],device=DEVICE)
            ps=torch.tensor(((support.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)[None].expand(len(f),-1)
            ys=torch.tensor((support.p_B-support.p_A).to_numpy(np.float32),device=DEVICE)[None].expand(len(f),-1)
            q=torch.tensor([[mapping[m]] for m in f.chemical_identity],device=DEVICE)
            qp=torch.tensor(((f.p_A.to_numpy()-6)/2).astype(np.float32)[:,None],device=DEVICE)
            mu,v=conditional_point(model,matrix[ids][None].expand(len(f),-1,-1),ps,ys,matrix[q],qp)
            target=torch.tensor((f.p_B-f.p_A).to_numpy(np.float32),device=DEVICE)
            vals.append(float(.5*(v.log()+(mu-target).square()/v).mean()))
    return np.mean(vals)


def main(seeds,kinds,steps):
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=[]
    path=RESULTS/'metrics.csv'
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.model.isin(kinds))].to_dict('records')
    for seed in seeds:
        cached,trainobs=point_cache(obs,seed,roots,mapping,strict_context=False,scaffold_salt=False)
        sp={s:partition(s,seed) for s in obs.scaffold.unique()}
        valobs=obs[obs.scaffold.map(sp).eq('validation')]
        trainids=[mapping[m] for m in trainobs.chemical_identity.unique()];dm=desc[trainids].mean(0);ds=desc[trainids].std(0).clip(.01)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-dm)/ds,-8,8)],1),device=DEVICE)
        groups={p:g for p,g in trainobs.groupby('pair_id')}
        split=read(f'anchor_split_scaffold_seed_{seed}.parquet')
        sar={p:data.merge(split[split.partition.eq(p)][['comparison_id']],on='comparison_id') for p in ['validation','test']}
        known=trainobs.groupby('pair_id').size();known=set(known[known>=30].index)
        sens={p:obs[obs.scaffold.map(sp).eq(p)&obs.pair_id.isin(known)].copy().reset_index(drop=True) for p in ['validation','test']}
        center=trainobs.groupby('pair_id').signed_delta.median()
        for f in sens.values():
            f['pair_offset']=f.pair_id.map(center);f['residual_abs']=abs(f.signed_delta-f.pair_offset);f['binary_loss']=(f.residual_abs>=.5).astype(int)
        for task,frames in [(1,sar),(2,sens)]:
            for p,f in frames.items():
                qs=set(f.scaffold_1)|set(f.scaffold_2) if task==1 else set(f.scaffold)
                assert set(trainobs.scaffold).isdisjoint(qs)
        print(seed,'point pairs',len(cached),'root train rows',len(trainobs),flush=True)
        for kind in kinds:
            start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=(NeuralResponseFieldGP(matrix.shape[1]) if kind=='neural_mean_gp' else ConditionalGP(matrix.shape[1])).to(DEVICE)
            if kind=='fixed_tanimoto_gp':
                for p in model.context.parameters():p.requires_grad_(False)
            opt=torch.optim.AdamW([
                {'params':[p for n,p in model.named_parameters() if n.startswith(('context.','mean_')) and p.requires_grad],'lr':.0007,'weight_decay':.003},
                {'params':[p for n,p in model.named_parameters() if not n.startswith(('context.','mean_'))],'lr':.01,'weight_decay':0.}])
            best=np.inf;state=None;best_step=0;hist=[]
            for step in range(steps):
                model.train();batch=point_batch(cached,matrix,rng,int(rng.choice([3,5,10])))
                mu,v=conditional_point(model,*batch[:-1]);target=batch[-1]
                loss=.5*(v.log()+(mu-target).square()/v).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%200==0:
                    model.eval();score=validation_point(model,valobs,groups,mapping,matrix)
                    hist.append({'step':step+1,'loss':float(loss.detach()),'validation_pair_macro_shift_NLL':score})
                    if score<best:best=score;state=copy.deepcopy(model.state_dict());best_step=step+1
                    if step+1-best_step>=800:break
            model.load_state_dict(state);model.eval()
            torch.save({'state_dict':model.state_dict(),'dim':matrix.shape[1],'desc_mean':dm,'desc_std':ds,'best_step':best_step,'kind':kind},RESULTS/f'{kind}_seed_{seed}.pt')
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            for task,frames in [(1,sar),(2,sens)]:
                for part,f in frames.items():
                    out=inference(model,f,trainobs,mapping,matrix,task);out.to_csv(RESULTS/f'task{task}_{kind}_{part}_seed_{seed}.csv',index=False)
                    rows.append({'seed':seed,'task':task,'model':kind,'partition':part,'best_step':best_step,**metric(f,out,task)})
            pd.DataFrame(rows).to_csv(path,index=False)
            print(seed,kind,'step',best_step,'seconds',round(time.monotonic()-start,1),'test',pd.DataFrame(rows).query("seed==@seed and model==@kind and partition=='test'")[['task','AP','AUROC','pair_macro_AUROC','macro_MAE']].round(4).to_dict('records'),flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps({'training':'All paired IC50 shift observations in global training scaffolds; no query scaffold in references; one shared signed field yields both task1 and task2',
        'selection':'Pair-macro held-out validation SHIFT NLL; no test/classification selection',
        'root_data':'Molecule shift episodes need3 references outside query scaffold; no analogue-similarity or A-gap eligibility',
        'evaluation':'Original task1 anchored split and task2 >=30 training molecule cohort unchanged; max100 label-blind historical references per context',
        'model':'One neural mean GP or one nonneural covariance GP; no model fusion', 'seeds':seeds,'steps':steps,
        'code_sha256':digest(Path(__file__))},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--models',nargs='+',default=['fixed_tanimoto_gp','neural_mean_gp']);p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args();main(a.seeds,a.models,a.steps)
