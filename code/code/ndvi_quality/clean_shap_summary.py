"""Summarize global and site-level SHAP contributions across three model seeds."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ndvi_clean_domain import affected_mask,integer_keys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--caches',nargs=3,type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--invalid-inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=integer_keys(pd.read_csv(a.manifest))
    if m.sample_id.duplicated().any(): raise ValueError('Duplicate sample IDs')
    if affected_mask(m,pd.read_csv(a.invalid_inputs)).any(): raise ValueError('Manifest still contains invalid windows')
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('Output directory must be empty')
    values=[];signed=[];sites=[];seeds=[];features=None
    for path in a.caches:
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['explained_sample_ids'],m.sample_id): raise ValueError('Sample IDs/order mismatch')
            names=list(z['feature_names']);v=z['shap_values_gC_m2_month']
            if v.shape!=(len(m),36,19) or not np.isfinite(v).all(): raise ValueError('Invalid SHAP tensor')
            if features is not None and names!=features: raise ValueError('Feature order mismatch')
            if np.asarray(z['window']).reshape(-1).tolist()!=[36]: raise ValueError('Expected W36')
            features=names;seed=int(np.asarray(z['model_seed']).item());seeds.append(seed)
            mag=abs(v).sum(axis=1);values.append(mag.mean(axis=0));signed.append(v.sum(axis=1).mean(axis=0))
            sf=pd.DataFrame(mag,columns=names).assign(point_id=m.point_id.to_numpy()).groupby('point_id').mean()
            sf['seed']=seed;sites.append(sf.reset_index())
    if sorted(seeds)!=[42,2024,3407]: raise ValueError('Expected three different model seeds')
    a.output.mkdir(parents=True,exist_ok=True)
    v=np.stack(values);s=np.stack(signed)
    summary=pd.DataFrame({'predictor':features,'mean_abs_shap':v.mean(axis=0),'seed_sd':v.std(axis=0,ddof=1),'signed_mean':s.mean(axis=0),'signed_seed_sd':s.std(axis=0,ddof=1)})
    summary.sort_values('mean_abs_shap',ascending=False).to_csv(a.output/'global_shap.csv',index=False,float_format='%.17g')
    pd.concat(sites).to_csv(a.output/'site_shap_by_seed.csv',index=False,float_format='%.17g')
    pd.concat(sites).drop(columns='seed').groupby('point_id').mean().to_csv(a.output/'site_shap_consensus.csv',float_format='%.17g')

if __name__=='__main__': main()
