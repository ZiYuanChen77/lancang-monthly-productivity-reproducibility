"""Build site-wise monthly histories and chronological sample keys."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
WINDOWS = (6, 12, 18, 24, 36, 48)
MAX_WINDOW = max(WINDOWS)
POINT_COLS = ('point_id', 'Lon', 'Lat', 'Year', 'Month')
DYNAMIC = ('LST_Day_1km', 'NDVI', 'Precipitation_mm', 'VPD', 'surface_solar_radiation_downwards_sum', 'temperature_2m', 'volumetric_soil_water_layer_1')
STATIC = ('elevation', 'slope')
VEGETATION = ('Veg_Coniferous', 'Veg_Shrub', 'Veg_Meadow_497', 'Veg_Meadow_499', 'Veg_Meadow_504', 'Veg_Sparse')
FEATURES = DYNAMIC + STATIC + ('Aspect_Sin', 'Aspect_Cos') + VEGETATION + ('Target_Month_Sin', 'Target_Month_Cos')
NDVI_SENTINEL = {'point_id': '233', 'Year': 2022, 'Month': 10, 'value': -9999.0}
VEGETATION_CODE = {61: 'Coniferous', 246: 'Shrub', 497: 'Meadow_497', 499: 'Meadow_499', 504: 'Meadow_504', 556: 'Sparse'}

def month_index(year: int, month: int) -> int:
    return int(year) * 12 + int(month)

def month_encoding(month: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * int(month) / 12.0
    return (float(np.sin(angle)), float(np.cos(angle)))

def build_frame(input_path: Path, max_forward_fill: int) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype={'point_id': str})
    required = set(POINT_COLS) | {'daima', 'aspect', 'NPP'} | set(DYNAMIC) | set(STATIC)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Input table is missing required columns: {missing}')
    for column in ('Lon', 'Lat', 'Year', 'Month', 'daima', 'aspect', 'NPP', *DYNAMIC, *STATIC):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['point_id'] = frame['point_id'].astype(str).str.replace('\\.0$', '', regex=True)
    sentinel_count = int((frame['point_id'].eq(NDVI_SENTINEL['point_id']) & frame['Year'].eq(NDVI_SENTINEL['Year']) & frame['Month'].eq(NDVI_SENTINEL['Month']) & frame['NDVI'].eq(NDVI_SENTINEL['value'])).sum())
    frame = frame.replace(-9999, np.nan)
    frame['Year'] = frame['Year'].astype('Int64')
    frame['Month'] = frame['Month'].astype('Int64')
    frame['daima'] = frame['daima'].astype('Int64')
    frame = frame.sort_values(['point_id', 'Year', 'Month']).reset_index(drop=True)
    if frame.duplicated(['point_id', 'Year', 'Month']).any():
        raise ValueError('Duplicate point_id-Year-Month records are not allowed.')
    if frame['daima'].isna().any():
        raise ValueError('Vegetation codes contain missing values.')
    unknown = sorted(set(frame['daima'].astype(int)) - set(VEGETATION_CODE))
    if unknown:
        raise ValueError(f'Unknown vegetation codes: {unknown}')
    frame[list(DYNAMIC)] = frame.groupby('point_id', sort=False)[list(DYNAMIC)].ffill(limit=max_forward_fill)
    if sentinel_count != 1:
        raise ValueError('The monthly input must contain exactly one NDVI=-9999 value at point_id 233, 2022-10.')
    sentinel_rows = frame['point_id'].eq(NDVI_SENTINEL['point_id']) & frame['Year'].eq(NDVI_SENTINEL['Year']) & frame['Month'].eq(NDVI_SENTINEL['Month'])
    if int(sentinel_rows.sum()) != 1:
        raise ValueError('NDVI sentinel key is not unique after sorting.')
    frame.loc[sentinel_rows, 'NDVI'] = NDVI_SENTINEL['value']
    frame['Aspect_Sin'] = np.sin(frame['aspect'] * np.pi / 180.0)
    frame['Aspect_Cos'] = np.cos(frame['aspect'] * np.pi / 180.0)
    veg = pd.get_dummies(frame['daima'].map(VEGETATION_CODE), prefix='Veg', dtype=np.float32).reindex(columns=VEGETATION, fill_value=0.0)
    frame = pd.concat([frame, veg], axis=1)
    frame['Target_Month_Sin'] = 0.0
    frame['Target_Month_Cos'] = 0.0
    return frame

def build_samples(frame: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_raw = {window: [] for window in WINDOWS}
    y_values: list[float] = []
    meta: list[list[float]] = []
    point_ids: list[str] = []
    for point_id, group in frame.groupby('point_id', sort=True):
        group = group.sort_values(['Year', 'Month']).reset_index(drop=True)
        if len(group) <= MAX_WINDOW:
            continue
        values = group[list(FEATURES)].to_numpy(dtype=np.float32)
        target = group['NPP'].to_numpy(dtype=np.float32)
        years = group['Year'].to_numpy(dtype=int)
        months = group['Month'].to_numpy(dtype=int)
        times = np.asarray([month_index(y, m) for y, m in zip(years, months)], dtype=int)
        lon = float(group['Lon'].dropna().iloc[0])
        lat = float(group['Lat'].dropna().iloc[0])
        for index in range(MAX_WINDOW, len(group)):
            if times[index] - times[index - MAX_WINDOW] != MAX_WINDOW:
                continue
            if not np.isfinite(target[index]):
                continue
            sin_m, cos_m = month_encoding(months[index])
            samples_for_target = []
            for window in WINDOWS:
                sample = values[index - window:index].copy()
                sample[:, -2] = sin_m
                sample[:, -1] = cos_m
                if not np.isfinite(sample).all():
                    break
                samples_for_target.append((window, sample))
            else:
                for window, sample in samples_for_target:
                    x_raw[window].append(sample)
                y_values.append(float(target[index]))
                meta.append([lon, lat, float(years[index]), float(months[index])])
                point_ids.append(str(point_id))
    arrays = {window: np.asarray(rows, dtype=np.float32) for window, rows in x_raw.items()}
    y_raw = np.asarray(y_values, dtype=np.float32)
    metadata = np.asarray(meta, dtype=np.float64)
    point_array = np.asarray(point_ids, dtype='<U32')
    if not len(y_raw) or any((array.shape[0] != len(y_raw) for array in arrays.values())):
        raise RuntimeError('No aligned samples were produced.')
    years = metadata[:, 2].astype(int)
    splits = np.where(years <= 2018, 'train', np.where(years <= 2021, 'val', 'test'))
    sample_index = pd.DataFrame({'sample_id': np.arange(len(y_raw), dtype=int), 'point_id': point_array, 'Lon': metadata[:, 0], 'Lat': metadata[:, 1], 'Year': years, 'Month': metadata[:, 3].astype(int), 'split': splits})
    return (arrays, y_raw, metadata, point_array, sample_index)
