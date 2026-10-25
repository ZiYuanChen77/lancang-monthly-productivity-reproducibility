"""Prepare W36 evaluation keys and filter NDVI-affected prediction and SHAP records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['point_id', 'Year', 'Month']


def integer_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in KEYS:
        values = pd.to_numeric(out[col], errors='raise')
        if values.isna().any() or (values != np.floor(values)).any():
            raise ValueError(f'Invalid integer key: {col}')
        out[col] = values.astype(np.int64)
    if not out['Month'].between(1, 12).all():
        raise ValueError('Month must be 1--12.')
    return out


def affected_mask(frame: pd.DataFrame, invalid_inputs: pd.DataFrame, window: int = 36) -> np.ndarray:
    """A target t is affected iff the invalid input belongs to [t-W, t-1]."""
    frame = integer_keys(frame)
    invalid_inputs = integer_keys(invalid_inputs)
    target_time = frame['Year'].to_numpy() * 12 + frame['Month'].to_numpy() - 1
    point_ids = frame['point_id'].to_numpy()
    affected = np.zeros(len(frame), dtype=bool)
    for row in invalid_inputs.itertuples(index=False):
        bad_time = row.Year * 12 + row.Month - 1
        distance = target_time - bad_time
        affected |= (point_ids == row.point_id) & (distance >= 1) & (distance <= window)
    return affected


def aligned_target_index(monthly: pd.DataFrame) -> pd.DataFrame:
    """Assign sample IDs after the 48-month alignment period."""
    monthly = integer_keys(monthly).sort_values(KEYS, kind='stable').reset_index(drop=True)
    parts = []
    for _, site in monthly.groupby('point_id', sort=True):
        times = site['Year'].to_numpy() * 12 + site['Month'].to_numpy() - 1
        if len(site) != 300 or not np.all(np.diff(times) == 1):
            raise ValueError('Expected complete 300-month site histories.')
        if not np.isfinite(site['NPP'].to_numpy(float)).all():
            raise ValueError('Targets must be finite.')
        part = site.iloc[48:][KEYS + ['Lon', 'Lat', 'daima']].copy()
        parts.append(part)
    index = pd.concat(parts, ignore_index=True)
    index.insert(0, 'sample_id', np.arange(len(index), dtype=np.int64))
    index['split'] = np.where(index['Year'] <= 2018, 'train', np.where(index['Year'] <= 2021, 'val', 'test'))
    return index


def prepare(source_root: Path, output: Path) -> dict:
    monthly = integer_keys(pd.read_csv(source_root / 'public_data/model_input/monthly_model_input.csv'))
    ndvi = monthly['NDVI'].to_numpy(float)
    # Negative NDVI itself is valid; detect only non-finite values and the sentinel.
    bad = monthly.loc[~np.isfinite(ndvi) | (ndvi == -9999), KEYS + ['NDVI']].copy()
    if bad[KEYS].to_records(index=False).tolist() != [(233, 2022, 10)]:
        raise ValueError('Expected the NDVI missing-input key (233, 2022, 10).')
    index = aligned_target_index(monthly)
    index['exclude_invalid_ndvi_history'] = affected_mask(index, bad)
    expected = {'train': 41496, 'val': 8892, 'test': 11856}
    if index['split'].value_counts().to_dict() != expected:
        raise ValueError('Unexpected Train/Validation/Evaluation record counts.')
    removed = index.loc[index['exclude_invalid_ndvi_history']]
    if len(removed) != 36 or set(removed['split']) != {'test'}:
        raise ValueError('Unexpected affected target windows.')
    output.mkdir(parents=True, exist_ok=True)
    bad.to_csv(output / 'invalid_ndvi_inputs.csv', index=False)
    index.to_csv(output / 'aligned_target_index_with_ndvi_flag.csv', index=False)
    removed.to_csv(output / 'excluded_evaluation_36.csv', index=False)
    clean = index.loc[(index['split'] == 'test') & ~index['exclude_invalid_ndvi_history']].copy()
    clean.to_csv(output / 'clean_evaluation_11820.csv', index=False)
    # Annual comparisons require all twelve months, even when monthly keys are valid.
    complete_keys = []
    for (point, year), group in clean.groupby(['point_id', 'Year'], sort=True):
        if set(group['Month']) == set(range(1, 13)) and len(group) == 12:
            complete_keys.append({'point_id': point, 'Year': year, 'n_months': 12})
    annual = pd.DataFrame(complete_keys)
    annual.to_csv(output / 'complete_site_years.csv', index=False)
    annual.loc[annual['Year'] == 2025].to_csv(output / 'complete_annual_sites_2025.csv', index=False)
    sample_root = source_root / 'code/data_documentation/shap_sample_keys/sample_keys'
    sampling = []
    for draw, expected_n, expected_sites in [('E0', 997, 247), ('E1', 996, 246), ('E2', 997, 247), ('E3', 997, 247)]:
        frame = integer_keys(pd.read_csv(sample_root / f'shap_explained_{draw}.csv'))
        if frame['sample_id'].duplicated().any():
            raise ValueError(f'Duplicate SHAP sample IDs in {draw}.')
        reconstructed = index.set_index('sample_id').loc[frame['sample_id'], KEYS].reset_index(drop=True)
        if not reconstructed.equals(frame[KEYS].reset_index(drop=True)):
            raise ValueError(f'Sample IDs disagree with the input manifest {draw}.')
        keep = ~affected_mask(frame, bad)
        retained = frame.loc[keep].copy()
        if len(retained) != expected_n or retained['point_id'].nunique() != expected_sites:
            raise ValueError(f'Unexpected cleaned SHAP manifest {draw}.')
        retained.to_csv(output / f'shap_explained_{draw}_clean.csv', index=False)
        frame.loc[~keep].to_csv(output / f'shap_explained_{draw}_excluded.csv', index=False)
        sampling.append({'draw': draw, 'original_n': len(frame), 'excluded_n': int((~keep).sum()), 'clean_n': len(retained), 'clean_sites': int(retained['point_id'].nunique())})
    background = integer_keys(pd.read_csv(sample_root / 'shap_background_manifests.csv'))
    if affected_mask(background, bad).any() or not (background['Year'] <= 2018).all():
        raise ValueError('Unexpected affected or non-training SHAP background.')
    pd.DataFrame(sampling).to_csv(output / 'shap_clean_sample_counts.csv', index=False)
    summary = {
        'data_commit': 'd877751e7c8e799db1e881a0e3f60a0a0b6a55e7',
        'original_split_counts': expected,
        'affected_train': 0, 'affected_validation': 0, 'affected_evaluation': 36,
        'clean_evaluation': len(clean), 'clean_sites': int(clean['point_id'].nunique()),
        'excluded_by_year': removed.groupby('Year').size().to_dict(),
        'complete_2025_sites': int((annual['Year'] == 2025).sum()),
        'shap_sample_counts': sampling, 'background_affected': False,
    }
    (output / 'domain_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def filter_predictions(path: Path, prepared: Path, output: Path) -> dict:
    """Filter prediction records using the W36 NDVI exclusion keys."""
    frame = integer_keys(pd.read_csv(path))
    index = integer_keys(pd.read_csv(prepared / 'aligned_target_index_with_ndvi_flag.csv'))
    expected = index.loc[index['split'] == 'test', KEYS]
    if 'seed' not in frame:
        raise ValueError('Provide explicit seed IDs (or deterministic) to preserve run identity.')
    if frame.duplicated(['seed'] + KEYS).any():
        raise ValueError('Duplicate prediction keys within seed.')
    expected_keys = set(map(tuple, expected.to_numpy()))
    for seed, part in frame.groupby('seed', dropna=False):
        if set(map(tuple, part[KEYS].to_numpy())) != expected_keys:
            raise ValueError(f'Incomplete or non-evaluation keys for seed {seed}.')
    bad = pd.read_csv(prepared / 'invalid_ndvi_inputs.csv')
    mask = affected_mask(frame, bad)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == path.resolve():
        raise ValueError('Refuse to overwrite the original predictions.')
    frame.loc[~mask].to_csv(output, index=False)
    return {'input': str(path), 'output': str(output), 'removed_rows': int(mask.sum()), 'retained_rows': int((~mask).sum())}


def filter_shap_cache(path: Path, manifest_path: Path, prepared: Path, output: Path) -> dict:
    """Drop all 36x19 contributions for affected samples; retain background IDs."""
    manifest = integer_keys(pd.read_csv(manifest_path))
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    required = {'shap_values_gC_m2_month', 'explained_sample_ids', 'background_sample_ids', 'feature_names', 'window', 'model_seed'}
    if not required.issubset(arrays):
        raise ValueError(f'Missing SHAP cache fields: {required - arrays.keys()}')
    ids = arrays['explained_sample_ids']
    values = arrays['shap_values_gC_m2_month']
    if not np.array_equal(ids, manifest['sample_id'].to_numpy()):
        raise ValueError('SHAP sample IDs/order must match the input manifest.')
    if values.shape != (len(manifest), 36, 19) or not np.isfinite(values).all():
        raise ValueError('Unexpected or non-finite SHAP tensor.')
    expected_features = ['LST_Day_1km', 'NDVI', 'Precipitation_mm', 'VPD', 'surface_solar_radiation_downwards_sum', 'temperature_2m', 'volumetric_soil_water_layer_1', 'elevation', 'slope', 'Aspect_Sin', 'Aspect_Cos', 'Veg_Coniferous', 'Veg_Shrub', 'Veg_Meadow_497', 'Veg_Meadow_499', 'Veg_Meadow_504', 'Veg_Sparse', 'Target_Month_Sin', 'Target_Month_Cos']
    if list(arrays['feature_names']) != expected_features or arrays['window'].tolist() != [36]:
        raise ValueError('Expected the W36 feature order.')
    bad = pd.read_csv(prepared / 'invalid_ndvi_inputs.csv')
    keep = ~affected_mask(manifest, bad)
    arrays['shap_values_gC_m2_month'] = values[keep]
    arrays['explained_sample_ids'] = ids[keep]
    # SHAP values and sample IDs share the explained-sample axis.
    if set(arrays) != required:
        raise ValueError('Expected the six-field SHAP cache schema.')
    if output.resolve() == path.resolve():
        raise ValueError('Refuse to overwrite the original SHAP cache.')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved['shap_values_gC_m2_month'], values[keep]):
            raise ValueError('Retained SHAP values changed on export.')
    record = {'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'output': str(output), 'before_n': len(ids), 'after_n': int(keep.sum()), 'excluded_sample_ids': ids[~keep].tolist(), 'retained_values_unchanged': True}
    output.with_suffix('.filter.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    pred = sub.add_parser('filter-predictions')
    pred.add_argument('--input', type=Path, required=True)
    pred.add_argument('--prepared', type=Path, required=True)
    pred.add_argument('--output', type=Path, required=True)
    shap = sub.add_parser('filter-shap')
    shap.add_argument('--input', type=Path, required=True)
    shap.add_argument('--manifest', type=Path, required=True)
    shap.add_argument('--prepared', type=Path, required=True)
    shap.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.source_root, args.output)
    elif args.action == 'filter-predictions':
        result = filter_predictions(args.input, args.prepared, args.output)
    else:
        result = filter_shap_cache(args.input, args.manifest, args.prepared, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
