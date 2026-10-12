"""Source- and scaffold-disjoint few-shot assay-transition GP pilot.

One GP with a shared/meta-learned molecular kernel; no prediction ensemble.
Each held-out context supplies permitted support labels. Query B labels are
used exclusively to score predictions. Active design sees query A and structure.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
import joblib

from prepare_sar_transfer_20261007 import OUT,SEEDS,partition,digest
from train_sar_transfer_baseline_20261007 import read,make_molecule_features

RESULTS=OUT/'models_meta_gp_transition_20261007'
RESULTS.mkdir(exist_ok=True)
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'


def stable_seed(*args):
    return int(hashlib.sha256('|'.join(map(str,args)).encode()).hexdigest()[:8],16)


def context_components(obs):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(x,y):
        a,b=find(x),find(y)
        if a!=b:parent[max(a,b)]=min(a,b)
    pairs=obs.drop_duplicates('pair_id')
    for row in pairs.itertuples():
        nodes=['d'+str(row.document_A),'d'+str(row.document_B),'a'+row.assay_A,'a'+row.assay_B]
        for node in nodes[1:]:union(nodes[0],node)
    return {r.pair_id:find('d'+str(r.document_A)) for r in pairs.itertuples()}


def split_data(obs,data,seed,root):
    cp={x:partition('ctx|'+x,seed) for x in set(root.values())}
    sp={x:partition('scf|'+x,seed) for x in obs.scaffold.unique()}
    pool=obs[obs.scaffold.map(sp).eq('train')].copy()
    counts=pool.groupby('pair_id').size();sc=pool.groupby(['pair_id','scaffold']).size()
    groups={k:g.reset_index(drop=True) for k,g in pool.groupby('pair_id')}
    parts={};episodes={};stats=[]
    for part in ['train','validation','test']:
        q=data[data.pair_id.map(root).map(cp).eq(part)&data.scaffold_1.map(sp).eq(part)&data.scaffold_2.map(sp).eq(part)].copy()
        support_n=q.pair_id.map(counts).fillna(0).to_numpy()
        if part=='train':
            n1=sc.reindex(pd.MultiIndex.from_arrays([q.pair_id,q.scaffold_1]),fill_value=0).to_numpy()
            n2=sc.reindex(pd.MultiIndex.from_arrays([q.pair_id,q.scaffold_2]),fill_value=0).to_numpy()
            support_n-=n1+np.where(q.scaffold_1.ne(q.scaffold_2),n2,0)
        q=q.loc[support_n>=10].reset_index(drop=True)
        parts[part]=q
        episodes[part]={p:g.reset_index(drop=True) for p,g in q.groupby('pair_id')}
        stats.append({'seed':seed,'partition':part,'queries':len(q),'assay_pairs':q.pair_id.nunique(),
                      'contexts':q.pair_id.map(root).nunique(),'targets':q.target_chembl_id.nunique(),
                      'losses':int(q.binary_loss.sum())})
    # Check context, assay, document, and global query scaffold separation.
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert set(parts[a].pair_id.map(root)).isdisjoint(set(parts[b].pair_id.map(root)))
        for paircols in [('assay_A','assay_B'),('document_A','document_B'),('scaffold_1','scaffold_2')]:
            assert (set(parts[a][paircols[0]])|set(parts[a][paircols[1]])).isdisjoint(
                set(parts[b][paircols[0]])|set(parts[b][paircols[1]]))
    train_pair_ids=set(parts['train'].pair_id)
    training_obs=pool[pool.pair_id.isin(train_pair_ids)]
    for part in ['validation','test']:
        assert set(training_obs.scaffold).isdisjoint(set(parts[part].scaffold_1)|set(parts[part].scaffold_2))
        for field in ['measurement_id','provenance_equivalence_groups']:
            def tokens(frame,columns):
                return {t for c in columns for s in frame[c].astype(str) for t in s.split(';')}
            left=tokens(training_obs,[field+'_A',field+'_B'])
            right=tokens(parts[part],[field+'_'+side+'_'+i for side in ['A','B'] for i in ['1','2']])
            assert left.isdisjoint(right),(seed,part,field)
    return parts,episodes,groups,training_obs,stats


class MetaGP(nn.Module):
    def __init__(self,dim,kind):
        super().__init__();self.kind=kind
        self.student=kind.endswith('_student');self.base_kind=kind.removesuffix('_student')
        if self.base_kind in ['deep','tanimoto_deep','tanimoto_deep_metric']:
            self.encoder=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,32))
        if self.student:self.raw_alpha=nn.Parameter(torch.tensor(math.log(math.expm1(2.))))
        self.log_offset=nn.Parameter(torch.tensor(math.log(3.)))
        self.log_slope=nn.Parameter(torch.tensor(math.log(.5)))
        self.log_amp=nn.Parameter(torch.tensor(math.log(.5)))
        self.log_noise=nn.Parameter(torch.tensor(math.log(.15)))
        self.log_length=nn.Parameter(torch.tensor(math.log(.7)))
        if self.base_kind.startswith('tanimoto_deep'):
            # Start close to the chemical kernel, rather than erase its
            # useful geometry with an untrained neural representation.
            with torch.no_grad():self.log_length.fill_(1.8)

    def noise(self):return self.log_noise.clamp(-4,1).exp().square()+1e-5

    def prior_shape(self):return 1.05+F.softplus(self.raw_alpha)

    def df(self,k):return 2*self.prior_shape()+k

    def kernel(self,x,p,y,q):
        offset=self.log_offset.clamp(-3,3).exp().square()
        slope=self.log_slope.clamp(-4,2).exp().square()
        out=offset+slope*p.unsqueeze(-1)*q.unsqueeze(-2)
        if self.base_kind=='linear':return out
        if self.base_kind in ['tanimoto','tanimoto_deep','tanimoto_deep_metric']:
            a=x[...,:2048];b=y[...,:2048]
            dot=a@b.transpose(-1,-2)
            chem=dot/(a.sum(-1,keepdim=True)+b.sum(-1).unsqueeze(-2)-dot).clamp_min(1e-6)
            if self.base_kind.startswith('tanimoto_deep'):
                a=F.normalize(self.encoder(x),dim=-1);b=F.normalize(self.encoder(y),dim=-1)
                distance=(a.square().sum(-1,keepdim=True)+b.square().sum(-1).unsqueeze(-2)-2*(a@b.transpose(-1,-2))).clamp_min(0)
                # Product of PSD kernels remains PSD. This is one process,
                # trained jointly, not a blend of fitted predictors.
                chem=chem*torch.exp(-distance/(2*self.log_length.clamp(-3,2).exp().square()))
        else:
            a=F.normalize(self.encoder(x),dim=-1);b=F.normalize(self.encoder(y),dim=-1)
            distance=(a.square().sum(-1,keepdim=True)+b.square().sum(-1).unsqueeze(-2)-2*(a@b.transpose(-1,-2))).clamp_min(0)
            chem=torch.exp(-distance/(2*self.log_length.clamp(-3,2).exp().square()))
        return out+self.log_amp.clamp(-4,2).exp().square()*chem

    def contrast(self,xs,ps,ys,xq,pq,gap_a):
        kss=self.kernel(xs,ps,xs,ps)
        kss=kss+(self.noise()+1e-4)*torch.eye(xs.shape[-2],device=xs.device)
        kqs=self.kernel(xq,pq,xs,ps)
        kqq=self.kernel(xq,pq,xq,pq)
        ch=torch.linalg.cholesky(kss)
        alpha=torch.cholesky_solve(ys.unsqueeze(-1),ch).squeeze(-1)
        cross=kqs[...,0,:]-kqs[...,1,:]
        mean=gap_a+(cross*alpha).sum(-1)
        prior=kqq[...,0,0]+kqq[...,1,1]-2*kqq[...,0,1]
        var=prior-(cross*torch.cholesky_solve(cross.unsqueeze(-1),ch).squeeze(-1)).sum(-1)+2*self.noise()
        var=var.clamp_min(1e-5)
        if self.student:
            # Integrate task scale s ~ InvGamma(a,a-1), including observation
            # noise. E[s]=1 a priori. Support outcomes update predictive scale.
            a=self.prior_shape();b=a-1+.5*(ys*alpha).sum(-1)
            var=var*b/(a+xs.shape[-2]/2-1)
        return mean,var.clamp_min(1e-5)


def molecule_matrix(obs):
    ids=sorted(obs.chemical_identity.unique());mapping={m:i for i,m in enumerate(ids)}
    mols=read('molecules.parquet').drop_duplicates('chemical_identity').set_index('chemical_identity')
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    f=np.zeros((len(ids),2048),np.float32)
    descriptors=make_molecule_features(set(ids))
    d=np.stack([descriptors[m][1] for m in ids])
    for i,m in enumerate(ids):
        mol=Chem.MolFromSmiles(mols.loc[m,'canonical_isomeric_smiles'])
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),f[i])
    return mapping,f,d


def query_arrays(frame,mapping):
    positive=frame.gap_A.to_numpy()>0
    strong=np.where(positive,frame.identity_1,frame.identity_2)
    weak=np.where(positive,frame.identity_2,frame.identity_1)
    ids=np.array([[mapping[a],mapping[b]] for a,b in zip(strong,weak)])
    p=np.column_stack([np.maximum(frame.p_A_1,frame.p_A_2),np.minimum(frame.p_A_1,frame.p_A_2)]).astype(np.float32)
    gap=frame.anchor_gap_abs.to_numpy(dtype=np.float32)
    target=(frame.gap_B*np.where(positive,1,-1)).to_numpy(dtype=np.float32)
    return ids,(p-6)/2,gap,target


def train_samples(episodes,groups,mapping):
    cached=[]
    for pair,frame in episodes.items():
        qids,p,gap,target=query_arrays(frame,mapping)
        pool=groups[pair]
        ids=np.array([mapping[i] for i in pool.chemical_identity]);pa=((pool.p_A-6)/2).to_numpy(np.float32)
        shift=(pool.p_B-pool.p_A).to_numpy(np.float32)
        eligible=[]
        for row in frame.itertuples():
            ix=np.flatnonzero((pool.scaffold!=row.scaffold_1)&(pool.scaffold!=row.scaffold_2))
            assert len(ix)>=10
            eligible.append(ix)
        cached.append((qids,p,gap,target,ids,pa,shift,eligible))
    return cached


def sample_batch(cached,matrix,rng,k,batch=32):
    qi=[];qp=[];ga=[];target=[];si=[];sp=[];sy=[]
    for _ in range(batch):
        c=cached[rng.integers(len(cached))];i=rng.integers(len(c[0]));s=rng.choice(c[7][i],k,replace=False)
        qi.append(c[0][i]);qp.append(c[1][i]);ga.append(c[2][i]);target.append(c[3][i])
        si.append(c[4][s]);sp.append(c[5][s]);sy.append(c[6][s])
    def tensor(a):return torch.as_tensor(np.array(a),device=DEVICE)
    return matrix[tensor(si)],tensor(sp),tensor(sy),matrix[tensor(qi)],tensor(qp),tensor(ga),tensor(target)


def evaluate_random(model,episodes,groups,mapping,matrix,seed,k=5,cap_queries=None):
    outputs=[];model.eval()
    with torch.no_grad():
        for pair,frame in episodes.items():
            if cap_queries is not None and len(frame)>cap_queries:
                frame=frame.sample(cap_queries,random_state=stable_seed(seed,pair,'q'))
            pool=groups[pair]
            rng=np.random.default_rng(stable_seed(seed,pair,'random'))
            support=pool.iloc[rng.permutation(len(pool))[:k]]
            ids,p,gap,target=query_arrays(frame,mapping)
            sidx=torch.tensor([mapping[i] for i in support.chemical_identity],device=DEVICE)
            xs=matrix[sidx][None].expand(len(frame),-1,-1)
            ps=torch.tensor(((support.p_A.to_numpy()-6)/2).astype(np.float32),device=DEVICE)[None].expand(len(frame),-1)
            ys=torch.tensor((support.p_B-support.p_A).to_numpy(np.float32),device=DEVICE)[None].expand(len(frame),-1)
            mean,var=model.contrast(xs,ps,ys,matrix[torch.tensor(ids,device=DEVICE)],
                                    torch.tensor(p,device=DEVICE),torch.tensor(gap,device=DEVICE))
            mean=mean.cpu().numpy();var=var.cpu().numpy()
            out=frame[['comparison_id','pair_id','target_chembl_id','binary_loss']].copy()
            out['gap_B_target']=target;out['gap_B_mean']=mean;out['gap_B_variance']=var
            if model.student:
                df=float(model.df(k));scale=np.sqrt(var*(df-2)/df)
                out['prediction']=student_t.cdf((.3-mean)/scale,df)
                out['nll']=-student_t.logpdf(target,df,loc=mean,scale=scale)
                out['halfwidth_90']=student_t.ppf(.95,df)*scale
            else:
                out['prediction']=ndtr((.3-mean)/np.sqrt(var))
                out['nll']=.5*(np.log(2*np.pi*var)+(target-mean)**2/var)
                out['halfwidth_90']=1.644854*np.sqrt(var)
            out['AE']=abs(target-mean)
            outputs.append(out)
    return pd.concat(outputs,ignore_index=True)


def summarize(frame):
    y=frame.binary_loss.to_numpy();p=frame.prediction.to_numpy()
    groups=[]
    for pair,g in frame.groupby('pair_id'):
        if len(g)>=5 and g.binary_loss.sum()>=2 and len(g)-g.binary_loss.sum()>=2:
            groups.append(roc_auc_score(g.binary_loss,g.prediction))
    return {'n':len(frame),'pairs':frame.pair_id.nunique(),'AP':float(average_precision_score(y,p)),
            'AUROC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'Brier':float(brier_score_loss(y,p)),'macro_MAE':float(frame.groupby('pair_id').AE.mean().mean()),
            'macro_NLL':float(frame.groupby('pair_id').nll.mean().mean()),
            'pair_macro_AUROC':float(np.mean(groups)) if groups else None,'macro_AUROC_pairs':len(groups),
            'coverage_90':float((abs(frame.gap_B_target-frame.gap_B_mean)<=
                (frame.halfwidth_90 if 'halfwidth_90' in frame else 1.644854*np.sqrt(frame.gap_B_variance))).mean())}


def fit(kind,seed,matrix,cached,val,groups,mapping,steps):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=MetaGP(matrix.shape[1],kind).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001 if model.base_kind in ['deep','tanimoto_deep'] else .015)
    if model.base_kind=='tanimoto_deep_metric':
        # Learn scalar calibration at the baseline rate, and regularise the
        # molecular metric at a slower rate. Avoid attributing scalar
        # underfitting to the representation's usefulness.
        optimizer=torch.optim.AdamW([
            {'params':list(model.encoder.parameters()),'lr':.0003,'weight_decay':.003},
            {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],
             'lr':.015,'weight_decay':0.}])
    best=float('inf');state=None;best_step=0;history=[]
    for step in range(steps):
        model.train();batch=sample_batch(cached,matrix,rng,int(rng.choice([3,5,10])))
        mean,var=model.contrast(*batch[:-1]);target=batch[-1]
        if model.student:
            df=model.df(batch[0].shape[-2]);scale=(var*(df-2)/df).sqrt()
            loss=-torch.distributions.StudentT(df,mean,scale).log_prob(target).mean()
        else:loss=.5*(var.log()+(target-mean).square()/var).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if (step+1)%100==0:
            pred=evaluate_random(model,val,groups,mapping,matrix,seed,5,64)
            score=summarize(pred)['macro_NLL']
            history.append({'step':step+1,'train_loss':float(loss.detach()),'val_macro_NLL':score})
            if score<best:
                best=score;state=copy.deepcopy(model.state_dict());best_step=step+1
            if step+1-best_step>=500:break
    model.load_state_dict(state);model.eval()
    return model,history,best_step


def main(seeds,kinds,steps):
    torch.set_num_threads(4)
    obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet')
    root=context_components(obs);mapping,bits,desc=molecule_matrix(obs)
    results=[];stats=[]
    if (RESULTS/'metrics.csv').exists():
        old=pd.read_csv(RESULTS/'metrics.csv');results=old.loc[~(old.seed.isin(seeds)&old.model.isin(kinds))].to_dict('records')
    if (RESULTS/'split_counts.csv').exists():
        old=pd.read_csv(RESULTS/'split_counts.csv');stats=old.loc[~old.seed.isin(seeds)].to_dict('records')
    for seed in seeds:
        parts,episodes,groups,trainobs,counts=split_data(obs,data,seed,root);stats.extend(counts)
        trainidx=np.array([mapping[x] for x in trainobs.chemical_identity.unique()])
        dm=desc[trainidx].mean(0);ds=desc[trainidx].std(0).clip(.01)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-dm)/ds,-8,8)],1),device=DEVICE)
        cached=train_samples(episodes['train'],groups,mapping)
        print('seed',seed,counts,flush=True)
        for part,frame in parts.items():
            frame[['comparison_id','pair_id','scaffold_1','scaffold_2']].to_csv(RESULTS/f'{part}_seed_{seed}.csv',index=False)
        for kind in kinds:
            start=time.monotonic();model,hist,step=fit(kind,seed,matrix,cached,episodes['validation'],groups,mapping,steps)
            torch.save({'kind':kind,'state_dict':model.state_dict(),'dim':matrix.shape[1],'desc_mean':dm,'desc_std':ds,
                        'best_step':step},RESULTS/f'{kind}_seed_{seed}.pt')
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            for part in ['validation','test']:
                for k in [1,3,5,10]:
                    pred=evaluate_random(model,episodes[part],groups,mapping,matrix,seed,k)
                    pred.to_csv(RESULTS/f'predictions_{kind}_{part}_k{k}_seed_{seed}.csv',index=False)
                    results.append({'seed':seed,'model':kind,'partition':part,'support_k':k,'best_step':step,
                                    'parameters':sum(p.numel() for p in model.parameters()),**summarize(pred)})
            pd.DataFrame(results).to_csv(RESULTS/'metrics.csv',index=False)
            print(kind,'seconds',round(time.monotonic()-start,1),'test k5',results[-2],flush=True)
        pd.DataFrame(stats).to_csv(RESULTS/'split_counts.csv',index=False)
        del matrix,cached
        if DEVICE.startswith('cuda'):torch.cuda.empty_cache()
    (RESULTS/'manifest.json').write_text(json.dumps({
        'task':'Source/assay-component and global-query-scaffold-disjoint few-shot SAR transfer',
        'source_sha256':digest(OUT/'anchor_transfer_all.parquet'),'observation_sha256':digest(OUT/'molecule_assay_observations.parquet'),
        'code_sha256':digest(Path(__file__)),'seeds':sorted({r['seed'] for r in results}),'max_steps':steps,
        'selection':'Validation pair-macro predictive contrast NLL (Gaussian or Student) at support K=5, 100-step checkpoints',
        'kernel':'Constant + linear assay-A potency + Tanimoto or learned RBF molecular residual; one process. Student variants integrate a per-task inverse-gamma covariance scale.',
        'support_policy':'Only train-scaffold reference compounds in each task; query contexts unseen in meta training; at least 10 eligible references',
        'label_note':'Likelihood fits median B-gap; original robust replicate-range loss used as classification evaluation, not exact latent event',
        'test_selection':'Random support order fixed by seed and pair, nested budgets; no B-based selection',
        'device':DEVICE},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--models',nargs='+',default=['linear','tanimoto','deep']);p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args();main(a.seeds,a.models,a.steps)
