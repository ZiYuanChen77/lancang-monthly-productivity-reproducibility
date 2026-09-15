"""Build aligned monthly sequence tensors for the W36 modelling workflow.

The script creates the six configured history lengths on a shared target-row
grid, derives the 19 model inputs, applies the chronological 2005--2018 /
2019--2021 / 2022--2025 split, and fits standardisation parameters on Train
target years. The recorded NDVI value at point 233 in October 2022 is retained
as part of the published modelling input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


WINDOWS = (6, 12, 18, 24, 36, 48)
MAX_WINDOW = max(WINDOWS)
POINT_COLS = ("point_id", "Lon", "Lat", "Year", "Month")
DYNAMIC = (
    "LST_Day_1km",
    "NDVI",
    "Precipitation_mm",
    "VPD",
    "surface_solar_radiation_downwards_sum",
    "temperature_2m",
    "volumetric_soil_water_layer_1",
)
STATIC = ("elevation", "slope")
VEGETATION = (
    "Veg_Coniferous",
    "Veg_Shrub",
    "Veg_Meadow_497",
    "Veg_Meadow_499",
    "Veg_Meadow_504",
    "Veg_Sparse",
)
FEATURES = DYNAMIC + STATIC + ("Aspect_Sin", "Aspect_Cos") + VEGETATION + (
    "Target_Month_Sin",
    "Target_Month_Cos",
)
SCALE_FEATURES = DYNAMIC + STATIC
RETAINED_NDVI_SENTINEL = {"point_id": "233", "Year": 2022, "Month": 10, "value": -9999.0}
VEGETATION_CODE = {
    61: "Coniferous",
    246: "Shrub",
    497: "Meadow_497",
    499: "Meadow_499",
    504: "Meadow_504",
    556: "Sparse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input monthly table (CSV).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for prepared tensors.")
    parser.add_argument(
        "--windows",
        default=",".join(map(str, WINDOWS)),
        help="Comma-separated history lengths; default is all configured lengths.",
    )
    parser.add_argument("--max-forward-fill", type=int, default=2)
    return parser.parse_args()


def month_index(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def month_encoding(month: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * int(month) / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


def parse_windows(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value not in WINDOWS for value in values):
        raise ValueError(f"windows must be chosen from {WINDOWS}")
    return tuple(dict.fromkeys(values))


def build_frame(input_path: Path, max_forward_fill: int) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype={"point_id": str})
    required = set(POINT_COLS) | {"daima", "aspect", "NPP"} | set(DYNAMIC) | set(STATIC)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    for column in ("Lon", "Lat", "Year", "Month", "daima", "aspect", "NPP", *DYNAMIC, *STATIC):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["point_id"] = frame["point_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Most -9999 codes are treated as missing before the bounded forward fill.
    # The disclosed evaluation-period NDVI sentinel is restored afterwards so
    # canonical-full W36 inputs preserve the canonical analysis definition.
    retained_sentinel_count = int((
        frame["point_id"].eq(RETAINED_NDVI_SENTINEL["point_id"])
        & frame["Year"].eq(RETAINED_NDVI_SENTINEL["Year"])
        & frame["Month"].eq(RETAINED_NDVI_SENTINEL["Month"])
        & frame["NDVI"].eq(RETAINED_NDVI_SENTINEL["value"])
    ).sum())
    frame = frame.replace(-9999, np.nan)
    frame["Year"] = frame["Year"].astype("Int64")
    frame["Month"] = frame["Month"].astype("Int64")
    frame["daima"] = frame["daima"].astype("Int64")
    frame = frame.sort_values(["point_id", "Year", "Month"]).reset_index(drop=True)

    if frame.duplicated(["point_id", "Year", "Month"]).any():
        raise ValueError("Duplicate point_id-Year-Month records are not allowed.")
    if frame["daima"].isna().any():
        raise ValueError("Vegetation codes contain missing values.")
    unknown = sorted(set(frame["daima"].astype(int)) - set(VEGETATION_CODE))
    if unknown:
        raise ValueError(f"Unknown vegetation codes: {unknown}")

    frame[list(DYNAMIC)] = frame.groupby("point_id", sort=False)[list(DYNAMIC)].ffill(limit=max_forward_fill)
    if retained_sentinel_count != 1:
        raise ValueError(
            "The monthly input must contain exactly one NDVI=-9999 value "
            "at point_id 233, 2022-10."
        )
    retained_sentinel = (
        frame["point_id"].eq(RETAINED_NDVI_SENTINEL["point_id"])
        & frame["Year"].eq(RETAINED_NDVI_SENTINEL["Year"])
        & frame["Month"].eq(RETAINED_NDVI_SENTINEL["Month"])
    )
    if int(retained_sentinel.sum()) != 1:
        raise ValueError("Retained NDVI sentinel key is not unique after sorting.")
    frame.loc[retained_sentinel, "NDVI"] = RETAINED_NDVI_SENTINEL["value"]
    frame["Aspect_Sin"] = np.sin(frame["aspect"] * np.pi / 180.0)
    frame["Aspect_Cos"] = np.cos(frame["aspect"] * np.pi / 180.0)
    veg = pd.get_dummies(
        frame["daima"].map(VEGETATION_CODE), prefix="Veg", dtype=np.float32
    ).reindex(columns=VEGETATION, fill_value=0.0)
    frame = pd.concat([frame, veg], axis=1)
    frame["Target_Month_Sin"] = 0.0
    frame["Target_Month_Cos"] = 0.0
    return frame


def build_samples(frame: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_raw = {window: [] for window in WINDOWS}
    y_values: list[float] = []
    meta: list[list[float]] = []
    point_ids: list[str] = []

    for point_id, group in frame.groupby("point_id", sort=True):
        group = group.sort_values(["Year", "Month"]).reset_index(drop=True)
        if len(group) <= MAX_WINDOW:
            continue
        values = group[list(FEATURES)].to_numpy(dtype=np.float32)
        target = group["NPP"].to_numpy(dtype=np.float32)
        years = group["Year"].to_numpy(dtype=int)
        months = group["Month"].to_numpy(dtype=int)
        times = np.asarray([month_index(y, m) for y, m in zip(years, months)], dtype=int)
        lon = float(group["Lon"].dropna().iloc[0])
        lat = float(group["Lat"].dropna().iloc[0])

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
    point_array = np.asarray(point_ids, dtype="<U32")
    if not len(y_raw) or any(array.shape[0] != len(y_raw) for array in arrays.values()):
        raise RuntimeError("No aligned samples were produced.")

    years = metadata[:, 2].astype(int)
    splits = np.where(years <= 2018, "train", np.where(years <= 2021, "val", "test"))
    sample_index = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_raw), dtype=int),
            "point_id": point_array,
            "Lon": metadata[:, 0],
            "Lat": metadata[:, 1],
            "Year": years,
            "Month": metadata[:, 3].astype(int),
            "split": splits,
        }
    )
    return arrays, y_raw, metadata, point_array, sample_index


def save_outputs(output_dir: Path, windows: tuple[int, ...], arrays: dict[int, np.ndarray], y_raw: np.ndarray,
                 metadata: np.ndarray, point_ids: np.ndarray, sample_index: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = sample_index["split"].to_numpy()
    train = split == "train"
    scale_indices = [FEATURES.index(name) for name in SCALE_FEATURES]
    scaler_x = StandardScaler().fit(arrays[MAX_WINDOW][train][:, :, scale_indices].reshape(-1, len(scale_indices)))
    scaler_y = StandardScaler().fit(y_raw[train].reshape(-1, 1))

    for window in windows:
        values = arrays[window].copy()
        block = values[:, :, scale_indices].reshape(-1, len(scale_indices))
        values[:, :, scale_indices] = scaler_x.transform(block).reshape(values[:, :, scale_indices].shape)
        np.save(output_dir / f"X_tensor_aligned_{window}.npy", values.astype(np.float32))
    y_scaled = scaler_y.transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)
    np.save(output_dir / "Y_tensor_aligned.npy", y_scaled)
    np.save(output_dir / "Y_tensor_raw.npy", y_raw)
    np.save(output_dir / "Meta_tensor_aligned.npy", metadata)
    np.save(output_dir / "PointID_tensor_aligned.npy", point_ids)
    np.save(output_dir / "Y_scaler_params.npy", np.asarray([scaler_y.mean_[0], scaler_y.scale_[0]], dtype=np.float64))
    np.savez(
        output_dir / "X_scaler_params.npz",
        mean=scaler_x.mean_.astype(np.float64),
        scale=scaler_x.scale_.astype(np.float64),
        columns=np.asarray(SCALE_FEATURES),
        feature_columns=np.asarray(FEATURES),
    )
    sample_index.to_csv(output_dir / "Sample_index_aligned.csv", index=False, encoding="utf-8")
    metadata_out = {
        "version": "prepared_month_inputs_v1",
        "output_dir": "data/processed",
        "max_window": MAX_WINDOW,
        "all_window_sizes_recommended": list(WINDOWS),
        "sample_count": int(len(y_raw)),
        "point_count": int(sample_index["point_id"].nunique()),
        "year_range": [int(sample_index["Year"].min()), int(sample_index["Year"].max())],
        "train_end_year": 2018,
        "val_end_year": 2021,
        "split_counts": sample_index["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).astype(int).to_dict(),
        "feature_names": list(FEATURES),
        "x_scaled_columns": list(SCALE_FEATURES),
        "y_scaler": {"mean": float(scaler_y.mean_[0]), "scale": float(scaler_y.scale_[0])},
        "notes": [
            "Target NPP is excluded from input features and is not interpolated.",
            "Dynamic predictors use forward fill with the configured maximum gap; the recorded NDVI sentinel is preserved.",
            "NDVI=-9999 at point_id 233, 2022-10 occurs in 36 W36 evaluation histories.",
            "X and y scalers are fitted on Train target years only.",
            "Target-month sine/cosine are deterministic calendar inputs.",
        ],
    }
    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    frame = build_frame(args.input, args.max_forward_fill)
    arrays, y_raw, metadata, point_ids, sample_index = build_samples(frame)
    save_outputs(args.output_dir, windows, arrays, y_raw, metadata, point_ids, sample_index)
    print(f"Wrote {len(y_raw)} aligned samples and windows {windows} to {args.output_dir}")


if __name__ == "__main__":
    main()
