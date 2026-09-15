"""Create the MOD17A2HGF.061 AppEEARS point input and request JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SITES = 247


def load_study_sites(path: Path) -> pd.DataFrame:
    sites = pd.read_csv(path, dtype={"point_id": "string"})
    required = {
        "point_id",
        "daima",
        "MOD17_sample_Lon",
        "MOD17_sample_Lat",
    }
    missing = sorted(required - set(sites.columns))
    if missing:
        raise ValueError(f"Study-site table is missing fields: {missing}")
    sites["point_id"] = sites["point_id"].str.replace(r"\.0$", "", regex=True).str.zfill(3)
    if len(sites) != EXPECTED_SITES or sites["point_id"].nunique() != EXPECTED_SITES:
        raise ValueError("The AppEEARS request requires 247 unique point_id values")
    if sites[["MOD17_sample_Lon", "MOD17_sample_Lat"]].isna().any(axis=None):
        raise ValueError("MOD17 sampled-pixel coordinates contain missing values")
    return sites.sort_values("point_id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-sites", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-name", default="lancang_MOD17A2HGF_247_sites")
    args = parser.parse_args()

    sites = load_study_sites(args.study_sites)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upload = pd.DataFrame(
        {
            "ID": sites["point_id"],
            "Category": sites["daima"].map(lambda value: f"daima_{int(value)}"),
            "Latitude": sites["MOD17_sample_Lat"],
            "Longitude": sites["MOD17_sample_Lon"],
        }
    )
    upload.to_csv(args.output_dir / "appeears_MOD17A2HGF_points.csv", index=False)

    request = {
        "task_type": "point",
        "task_name": args.task_name,
        "params": {
            "dates": [{"startDate": "12-26-2000", "endDate": "12-31-2025"}],
            "layers": [
                {"product": "MOD17A2HGF.061", "layer": "PsnNet_500m"},
                {"product": "MOD17A2HGF.061", "layer": "Psn_QC_500m"},
            ],
            "coordinates": [
                {
                    "id": row.point_id,
                    "category": f"daima_{int(row.daima)}",
                    "latitude": float(row.MOD17_sample_Lat),
                    "longitude": float(row.MOD17_sample_Lon),
                }
                for row in sites.itertuples(index=False)
            ],
        },
    }
    (args.output_dir / "appeears_MOD17A2HGF_request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    print(f"Prepared AppEEARS request for {len(sites)} sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
