# Input data

The primary input is `public_data_v1.1/model_input/monthly_model_input.csv`, with one row per `point_id`–`Year`–`Month` for 247 sites from 2001 through 2025. Its columns are:

`point_id`, `Lon`, `Lat`, `Year`, `Month`, `daima`, `aspect`, `NPP`, `LST_Day_1km`, `NDVI`, `Precipitation_mm`, `VPD`, `surface_solar_radiation_downwards_sum`, `temperature_2m`, `volumetric_soil_water_layer_1`, `elevation`, and `slope`.

The preprocessing code derives 19 model inputs: seven historical dynamic variables, elevation, slope, sine/cosine aspect, six vegetation indicators and sine/cosine target-month terms. Dynamic histories use months `t-36` through `t-1` for the canonical W36 model.

`public_data_v1.1/model_input/study_sites_247.csv` provides the fixed site identity and coordinates. `unseen_site_fold_assignment.csv` maps the same 247 point identifiers to five coordinate-clustered holdout folds with 36, 44, 34, 74 and 59 sites.

The chronological target-year split is:

- Train: 2005–2018
- Validation: 2019–2021
- Evaluation: 2022–2025

The data dictionary and source list are provided in `public_data_v1.1/documentation/`.

## Product identifiers

| Variable or role | Product |
|---|---|
| Monthly productivity | `MODIS/061/MOD17A2HGF`, `PsnNet` |
| MOD17 quality | AppEEARS `MOD17A2HGF.061`, `PsnNet_500m`, `Psn_QC_500m` |
| Annual reference | `MODIS/061/MOD17A3HGF`, `Npp`, `Npp_QC` |
| Land-surface temperature | `MODIS/061/MOD11A2`, `LST_Day_1km` |
| Vegetation index | `MODIS/061/MOD13A1`, `NDVI` |
| Climate | `ECMWF/ERA5_LAND/MONTHLY_AGGR` |
| Terrain | `COPERNICUS/DEM/GLO30` |

## Water–energy analysis inputs

The GAMM module expects the study-specific analysis-ready water–energy table and supporting baseline/quality tables listed in `code/water_energy_analysis/README.md`. These are part of the public-data assembly for full reproduction of the water–energy analyses.
