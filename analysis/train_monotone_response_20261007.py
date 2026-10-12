"""Train reference-conditioned potency-axis transport inside one GP.

Pilot runs are validation-only. Test evaluation requires --evaluate-test.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from meta_gp_sar_20261007 import (read,context_components,split_data,molecule_matrix,train_samples,
    sample_batch,SEEDS,DEVICE,RESULTS as BASE)
from conditional_neural_gp_20261007 import evaluate,design
from neural_response_field_20261007 import point_cache
from monotone_response_gp_20261007 import MonotoneResponseGP
from meta_gp_sar_20261007 import summarize
from prepare_sar_transfer_20261007 import OUT,digest
from sklearn.metrics import average_precision_score
RESULTS=OUT/'models_monotone_response_20261007';RESULTS.mkdir(exist_ok=True)


def fast_point_pools(cached,budgets=(3,5,10)):
    pools={k:[] for k in budgets}
    for c in cached:
        for k in pools:
            q=np.array([i for i in c[4] if len(c[3][i])>=k])
            if len(q):pools[k].append((c,q))
    return pools


def fast_point_batch(pools,matrix,rng,k,batch=32):
    valid=pools[k];si=[];sp=[];sy=[];qi=[];qp=[];target=[]
    for _ in range(batch):
        c,q=valid[rng.integers(len(valid))];i=int(rng.choice(q));s=rng.choice(c[3][i],k,replace=False)
        si.append(c[0][s]);sp.append(c[1][s]);sy.append(c[2][s]);qi.append([c[0][i]]);qp.append([c[1][i]]);target.append(c[2][i])
    def t(x):return torch.as_tensor(np.array(x),device=matrix.device)
    return matrix[t(si)],t(sp),t(sy),matrix[t(qi)],t(qp),t(target)


def fit(seed,kind,matrix,cached,points,episodes,groups,mapping,steps):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if kind=='heteroscedastic_monotone':
        from heteroscedastic_response_gp_20261007 import HeteroscedasticResponseGP
        model=HeteroscedasticResponseGP(matrix.shape[1]).to(DEVICE)
    else:model=MonotoneResponseGP(matrix.shape[1],conditional=kind=='conditional_monotone').to(DEVICE)
    base=torch.load(BASE/f'tanimoto_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(base['state_dict'],strict=False)
    opt=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if n.startswith(('context.','mean_warp.','variance_network.')) and p.requires_grad], 'lr':.0007,'weight_decay':.003},
        {'params':[p for n,p in model.named_parameters() if not n.startswith(('context.','mean_warp.','variance_network.')) and p.requires_grad],'lr':.003,'weight_decay':0.}])
    pools=fast_point_pools(points);best=-1.;best_step=0;state=None;history=[]
    for step in range(steps):
        model.train();k=int(rng.choice([3,5,10]));before=copy.deepcopy(rng.bit_generator.state)
        batch=sample_batch(cached,matrix,rng,k);after=copy.deepcopy(rng.bit_generator.state);rng.bit_generator.state=before
        labels=[]
        for _ in range(32):
            c=cached[rng.integers(len(cached))];i=rng.integers(len(c[0]));rng.choice(c[7][i],k,replace=False);labels.append(c[8][i])
        rng.bit_generator.state=after
        mean,v=model.contrast(*batch[:-1]);prob=.5*(1+torch.erf((.3-mean)/torch.sqrt(2*v)))
        loss=.5*(v.log()+(batch[-1]-mean).square()/v).mean()+.5*F.binary_cross_entropy(prob.clamp(1e-5,1-1e-5),torch.tensor(labels,device=DEVICE,dtype=torch.float32))
        pb=fast_point_batch(pools,matrix,rng,k);pm,pv=model.point(*pb[:-1])
        loss=loss+.5*.5*(pv.log()+(pb[-1]-pm).square()/pv).mean()
        # Identity-map regularisation protects extrapolation from few supports.
        slopes=model.warp_parameters(pb[1],pb[2])[2]
        loss=loss+.005*slopes.log().square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if (step+1)%100==0:
            model.eval();pred=[evaluate(model,episodes['validation'],groups,mapping,matrix,seed,5,d) for d in range(3)]
            ap=float(np.mean([average_precision_score(f.binary_loss,f.prediction) for f in pred]));nll=float(np.mean([summarize(f)['macro_NLL'] for f in pred]))
            history.append({'step':step+1,'loss':float(loss.detach()),'validation_AP':ap,'validation_macro_NLL':nll})
            print(seed,kind,step+1,'validation AP',round(ap,4),'NLL',round(nll,4),flush=True)
            if ap>best+1e-5:best=ap;state=copy.deepcopy(model.state_dict());best_step=step+1
            if step+1-best_step>=500:break
    model.load_state_dict(state);model.eval();return model,history,best_step


def main(seeds,kinds,steps,evaluate_test):
    torch.set_num_threads(4);obs=read('molecule_assay_observations.parquet');data=read('anchor_transfer_all.parquet');roots=context_components(obs)
    mapping,bits,desc=molecule_matrix(obs);rows=[];path=RESULTS/'metrics.csv'
    if path.exists():
        old=pd.read_csv(path);rows=old[~(old.seed.isin(seeds)&old.model.isin(kinds))].to_dict('records')
    for seed in seeds:
        parts,episodes,groups,_,_=split_data(obs,data,seed,roots)
        base=torch.load(BASE/f'tanimoto_seed_{seed}.pt',map_location=DEVICE,weights_only=False)
        matrix=torch.tensor(np.concatenate([bits,np.clip((desc-base['desc_mean'])/base['desc_std'],-8,8)],1),device=DEVICE)
        cached=train_samples(episodes['train'],groups,mapping)
        cached=[c+(f.binary_loss.to_numpy(np.float32),) for c,f in zip(cached,episodes['train'].values())]
        points,train=point_cache(obs,seed,roots,mapping)
        from audit_response_field_20261007 import separation
        for part in ['validation','test']:
            separation(train,parts[part],1,seed,part)
            assert set(train.pair_id.map(roots)).isdisjoint(set(parts[part].pair_id.map(roots)))
        for kind in kinds:
            start=time.monotonic();model,hist,best=fit(seed,kind,matrix,cached,points,episodes,groups,mapping,steps)
            pd.DataFrame(hist).to_csv(RESULTS/f'history_{kind}_seed_{seed}.csv',index=False)
            torch.save({'state_dict':model.state_dict(),'dim':matrix.shape[1],'best_step':best,'desc_mean':base['desc_mean'],'desc_std':base['desc_std']},RESULTS/f'{kind}_seed_{seed}.pt')
            for draw in range(3):
                pred=evaluate(model,episodes['validation'],groups,mapping,matrix,seed,5,draw)
                pred.to_csv(RESULTS/f'predictions_{kind}_validation_k5_seed_{seed}_draw{draw}.csv',index=False)
                rows.append({'model':kind,'seed':seed,'draw':draw,'partition':'validation','policy':'random','support_k':5,'best_step':best,**summarize(pred)})
                if evaluate_test:
                    outputs,_=design(model,episodes['test'],groups,mapping,bits,matrix,seed,draw)
                    for (policy,k),pred in outputs.items():
                        pred.to_csv(RESULTS/f'predictions_{kind}_test_{policy}_k{k}_seed_{seed}_draw{draw}.csv',index=False)
                        rows.append({'model':kind,'seed':seed,'draw':draw,'partition':'test','policy':policy,'support_k':k,'best_step':best,**summarize(pred)})
            pd.DataFrame(rows).to_csv(path,index=False);print(seed,kind,'finished',round(time.monotonic()-start,1),'seconds',flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps({'setting':'Original strict source/scaffold-disjoint few-shot cohort',
        'mean':'Neural reference-moment network sets9 positive slopes of a smooth monotone potency-axis map; one chemical GP residual',
        'control':'global_monotone learns9 global slopes and freezes neural conditioning; same scalar GP warm-start and root labels',
        'selection':'Validation AP,mean3 random support draws,K5; no test selection','test_evaluated':evaluate_test,
        'seeds':seeds,'models':kinds,'max_steps':steps,'code_sha256':digest(Path(__file__)),
        'model_code_sha256':digest(Path(__file__).with_name('monotone_response_gp_20261007.py'))},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--models',nargs='+',default=['global_monotone','conditional_monotone']);p.add_argument('--steps',type=int,default=2000);p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args();main(a.seeds,a.models,a.steps,a.evaluate_test)
