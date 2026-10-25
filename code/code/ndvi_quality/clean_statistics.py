"""Calculate clean-domain evaluation diagnostics and clustered intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ndvi_clean_domain import integer_keys, affected_mask

KEYS = ['point_id', 'Year', 'Month']
SEEDS = [42, 2024, 3407]

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p-y
    sst = np.sum((y-y.mean())**2)
    return dict(RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(abs(e)),
                R2=1-np.sum(e**2)/sst if sst else np.nan, Bias=e.mean(),
                Pearson_r=np.corrcoef(y,p)[0,1] if y.std() and p.std() else np.nan)

def read(path, prepared, evaluation_keys=None, arm=None):
    f = integer_keys(pd.read_csv(path))
    if 'fold_id' not in f and 'fold' in f: f['fold_id'] = f['fold']
    if arm is not None:
        if 'arm' not in f: raise ValueError('An arm was requested but the input has no arm column')
        f = f.loc[f.arm.eq(arm)].copy()
    elif 'arm' in f and f.arm.nunique() > 1:
        raise ValueError('Select one arm with --arm or --paired-arm')
    for old, new in [('y_true_gC_m2_month','y_true'), ('y_pred_gC_m2_month','y_pred')]:
        if new not in f and old in f: f[new] = f[old]
    required = KEYS+['seed','y_true','y_pred']
    if not set(required).issubset(f): raise ValueError(f'Required: {required}')
    f['seed'] = pd.to_numeric(f.seed, errors='raise')
    if sorted(f.seed.unique()) != SEEDS: raise ValueError('Expected seeds 42, 2024, 3407')
    if f.duplicated(['seed']+KEYS).any(): raise ValueError('Duplicate seed/site/month')
    if not np.isfinite(f[['y_true','y_pred']].to_numpy(float)).all(): raise ValueError('Nonfinite pairs')
    bad = pd.read_csv(prepared/'invalid_ndvi_inputs.csv')
    f = f.loc[~affected_mask(f,bad)].copy()
    expected = pd.read_csv(prepared/'clean_evaluation_11820.csv')
    expected_keys = set(map(tuple,expected[KEYS].to_numpy()))
    if evaluation_keys is not None:
        cohort = integer_keys(pd.read_csv(evaluation_keys))
        selected_keys = set(map(tuple,cohort[KEYS].to_numpy()))
        if not selected_keys or cohort.duplicated(KEYS).any() or not selected_keys.issubset(expected_keys):
            raise ValueError('Evaluation keys must be a unique nonempty subset of the W36 clean domain')
        expected_keys = selected_keys
        f = f.loc[[tuple(k) in expected_keys for k in f[KEYS].to_numpy()]].copy()
    for seed,part in f.groupby('seed'):
        if set(map(tuple,part[KEYS].to_numpy())) != expected_keys: raise ValueError(f'Incomplete clean cohort: {seed}')
    reference = f.pivot(index=KEYS,columns='seed',values='y_true')
    if not np.allclose(reference.to_numpy(),reference.to_numpy()[:,[0]],rtol=0,atol=0):
        raise ValueError('References differ between seeds')
    return f.sort_values(['seed']+KEYS).reset_index(drop=True)

def summarize(f, group):
    rows=[]
    for key,g in f.groupby(group,sort=True):
        vals = pd.DataFrame([metrics(p.y_true,p.y_pred) for _,p in g.groupby('seed')])
        first=g[g.seed==42]
        row={group:key,'records_per_seed':len(first),'sites':first.point_id.nunique()}
        for name in vals:
            row[name+'_mean']=vals[name].mean();row[name+'_sample_sd']=vals[name].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

def bootstrap(f, other, reps, seed, site_year):
    keys=KEYS
    if other is not None:
        other=other.sort_values(['seed']+keys).reset_index(drop=True)
        if not f[['seed']+keys].equals(other[['seed']+keys]): raise ValueError('Paired cohorts differ')
    parts=[f[f.seed==s].reset_index(drop=True) for s in SEEDS]
    paired=[other[other.seed==s].reset_index(drop=True) for s in SEEDS] if other is not None else None
    cluster_cols=['point_id','Year'] if site_year else ['point_id']
    groups=list(parts[0].groupby(cluster_cols,sort=True).indices.values())
    rng=np.random.default_rng(seed)
    draws=[]
    def estimate(indices):
        vals=pd.DataFrame([metrics(p.y_true.to_numpy()[indices],p.y_pred.to_numpy()[indices]) for p in parts]).mean()
        if paired is not None:
            vals=pd.DataFrame([metrics(p.y_true.to_numpy()[indices],p.y_pred.to_numpy()[indices]) for p in paired]).mean()-vals
        return vals
    point=estimate(np.arange(len(parts[0])))
    for _ in range(reps):
        idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        draws.append(estimate(idx))
    samples=pd.DataFrame(draws)
    return pd.DataFrame({'metric':point.index,'point_estimate':point.values,
        'ci_lower':samples.quantile(.025).values,'ci_upper':samples.quantile(.975).values,
        'contrast':'paired input minus primary input' if paired is not None else 'primary input',
        'cluster':'point_id-Year' if site_year else 'point_id','replicates':reps})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--paired',type=Path)
    p.add_argument('--evaluation-keys',type=Path,help='Optional common-domain keys for paired comparisons')
    p.add_argument('--arm',help='Select one arm from a spatial prediction table')
    p.add_argument('--paired-arm',help='Select the paired arm from a spatial prediction table')
    p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--random-seed',type=int,default=20240724)
    p.add_argument('--site-year',action='store_true')
    p.add_argument('--residual-seed',type=int,choices=SEEDS,help='Export residuals for an individual model seed.')
    a=p.parse_args()
    if a.bootstrap<0 or a.bootstrap==1: p.error('Bootstrap must be zero or at least two')
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('Output directory must be empty')
    f=read(a.input,a.prepared,a.evaluation_keys,a.arm);a.output.mkdir(parents=True,exist_ok=True)
    f.to_csv(a.output/'clean_predictions.csv',index=False,float_format='%.17g')
    rows=[dict(seed=s,**metrics(g.y_true,g.y_pred)) for s,g in f.groupby('seed')]
    pd.DataFrame(rows).to_csv(a.output/'metrics_by_seed.csv',index=False,float_format='%.17g')
    f['all']='all'
    ref=f[f.seed==42][KEYS+['y_true']].sort_values(['y_true']+KEYS,kind='stable')
    ref['decile']=np.arange(len(ref))*10//len(ref)+1
    f=f.merge(ref[KEYS+['decile']],on=KEYS,validate='many_to_one')
    for group in ['all','Month','Year','decile']+(['fold_id'] if 'fold_id' in f else []):
        summarize(f,group).to_csv(a.output/f'{group}_summary.csv',index=False,float_format='%.17g')
    first=f[f.seed==42][KEYS+['y_true']]
    predictions=f.pivot(index=KEYS,columns='seed',values='y_pred').mean(axis=1).rename('prediction').reset_index()
    ensemble=first.merge(predictions,on=KEYS,validate='one_to_one')
    annual=ensemble.groupby(['point_id','Year']).filter(lambda g:len(g)==12 and set(g.Month)==set(range(1,13)))
    annual=annual.groupby(['point_id','Year'],as_index=False)[['y_true','prediction']].sum()
    annual['residual']=annual.prediction-annual.y_true
    annual.to_csv(a.output/'complete_annual_site_values.csv',index=False,float_format='%.17g')
    annual.groupby('Year')[['y_true','prediction','residual']].mean().to_csv(a.output/'annual_site_means.csv',float_format='%.17g')
    residuals=f[['seed']+KEYS+['y_true','y_pred']].assign(residual=lambda g:g.y_pred-g.y_true)
    residuals.to_csv(a.output/'residuals_by_seed.csv',index=False,float_format='%.17g')
    if a.residual_seed is not None:
        residuals.loc[residuals.seed==a.residual_seed,KEYS+['y_true','y_pred','residual']].to_csv(a.output/f'residuals_seed_{a.residual_seed}.csv',index=False,float_format='%.17g')
    if a.bootstrap:
        other=read(a.paired,a.prepared,a.evaluation_keys,a.paired_arm) if a.paired else None
        bootstrap(f.sort_values(['seed']+KEYS).reset_index(drop=True),other,a.bootstrap,a.random_seed,a.site_year).to_csv(a.output/'bootstrap.csv',index=False,float_format='%.17g')
    (a.output/'run.json').write_text(json.dumps({'input':str(a.input),'paired':str(a.paired) if a.paired else None,'evaluation_keys':str(a.evaluation_keys) if a.evaluation_keys else None,'arm':a.arm,'paired_arm':a.paired_arm,'clean_records_per_seed':len(f.loc[f.seed==42]),'seeds':SEEDS,'bootstrap':a.bootstrap,'random_seed':a.random_seed,'residual_seed':a.residual_seed},indent=2),encoding='utf-8')

if __name__=='__main__': main()
