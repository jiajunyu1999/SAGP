"""Continuous transition profiles from permitted historical reference data.

References use global train scaffolds. Training-query scaffold(s) are excluded
before ALL summary calculations. Validation/test query B values are never read.
"""
import numpy as np
from scipy.special import ndtr
from prepare_sar_transfer_20261007 import partition


def reference_profiles(frames,obs,seed,task):
    mapping={s:partition(s,seed) for s in obs.scaffold.unique()}
    pool=obs[obs.scaffold.map(mapping).eq('train')]
    groups={p:g for p,g in pool.groupby('pair_id')};out={};cache={}
    for part,frame in frames.items():
        features=[];risk=[];mean=[]
        for row in frame.itertuples(index=False):
            exclude=(row.scaffold_1,row.scaffold_2) if task==1 else (row.scaffold,)
            if part!='train':exclude=()
            key=(row.pair_id,tuple(sorted(set(exclude))))
            if key not in cache:
                refs=groups.get(row.pair_id)
                if refs is not None and exclude:refs=refs[~refs.scaffold.isin(exclude)]
                n=0 if refs is None else len(refs)
                if n>=3:
                    pa=refs.p_A.to_numpy();shift=refs.signed_delta.to_numpy();pm=pa.mean();ym=shift.mean();med=np.median(shift)
                    vpa=np.mean((pa-pm)**2);vy=np.mean((shift-ym)**2);cov=np.mean((pa-pm)*(shift-ym))
                    slope=cov/(vpa+.05)
                    residual=shift-ym-slope*(pa-pm)
                    noise=max(.05,float(np.sqrt(np.mean(residual**2))))
                    values=[np.log1p(n),pm,np.sqrt(vpa),ym,med,np.sqrt(vy),slope,noise,
                            np.median(abs(shift-med)),pa.min(),pa.max(),np.mean(refs.max_B-refs.min_B),1.]
                else:values=[np.log1p(n),6.,1.,0.,0.,1.,0.,1.,.5,4.,9.,0.,0.]
                cache[key]=np.array(values,np.float32)
            v=cache[key];pm,ym,med,slope,noise=v[1],v[3],v[4],v[6],v[7]
            if task==1:
                strong=max(row.p_A_1,row.p_A_2);weak=min(row.p_A_1,row.p_A_2)
                gap=strong-weak;mb=(1+slope)*gap;sd=np.sqrt(2)*noise
                features.append(np.concatenate([v,[(strong-pm)/max(v[2],.1),(weak-pm)/max(v[2],.1),mb,sd]]))
                risk.append(ndtr((.3-mb)/sd));mean.append(mb)
            else:
                mu=ym+slope*(row.p_A-pm)-med
                features.append(np.concatenate([v,[(row.p_A-pm)/max(v[2],.1),mu,abs(mu)/noise]]))
                risk.append(ndtr((-.5-mu)/noise)+ndtr((mu-.5)/noise))
                mean.append(noise*np.sqrt(2/np.pi)*np.exp(-.5*(mu/noise)**2)+mu*(2*ndtr(mu/noise)-1))
        out[part]={'features':np.asarray(features,np.float32),'risk':np.asarray(risk,np.float32),'mean':np.asarray(mean,np.float32)}
    return out
