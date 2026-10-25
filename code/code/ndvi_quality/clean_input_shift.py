"""Summarize Training-to-Evaluation predictor distribution shifts for W36 inputs."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from ndvi_clean_domain import integer_keys,affected_mask

def wasserstein_distance(a,b):
    a=np.sort(np.asarray(a,float));b=np.sort(np.asarray(b,float))
    grid=np.sort(np.concatenate([a,b]))
    cdf_a=np.searchsorted(a,grid[:-1],side='right')/len(a)
    cdf_b=np.searchsorted(b,grid[:-1],side='right')/len(b)
    return np.sum(abs(cdf_a-cdf_b)*np.diff(grid))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['tensor','index','metadata','invalid-inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();idx=integer_keys(pd.read_csv(a.index));x=np.load(a.tensor,mmap_mode='r')
    features=json.loads(a.metadata.read_text(encoding='utf-8'))['feature_names']
    if x.shape!=(len(idx),36,len(features)):raise ValueError('Tensor/index/feature dimensions disagree')
    train=idx.split.eq('train').to_numpy();ev=idx.split.eq('test').to_numpy()
    clean=ev & ~affected_mask(idx,pd.read_csv(a.invalid_inputs))
    rows=[]
    for view in ['lag-1','W36 pooled']:
        for k,name in enumerate(features[:7]):
            t=np.asarray(x[train,-1,k] if view=='lag-1' else x[train,:,k].ravel(),float)
            e=np.asarray(x[clean,-1,k] if view=='lag-1' else x[clean,:,k].ravel(),float)
            if not np.isfinite(t).all() or not np.isfinite(e).all():raise ValueError('Nonfinite clean input')
            q01,q25,q75,q99=np.quantile(t,[.01,.25,.75,.99]);iqr=q75-q25
            if iqr<=0:raise ValueError('Zero Training IQR')
            median=(np.median(e)-np.median(t))/iqr;distance=wasserstein_distance(t,e)/iqr
            outside=((e<q01)|(e>q99)).mean()
            label='limited' if abs(median)<.25 and distance<.25 and outside<.05 else ('moderate' if abs(median)<.5 and distance<.5 and outside<.1 else 'pronounced')
            rows.append(dict(predictor=name,view=view,training_n=len(t),evaluation_n=len(e),normalized_median_shift=median,normalized_wasserstein=distance,outside_training_P1_P99_proportion=outside,descriptive_shift_class=label))
    if a.output.exists():raise ValueError('Output already exists')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False,float_format='%.17g')

if __name__=='__main__':main()
