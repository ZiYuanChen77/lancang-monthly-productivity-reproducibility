# Data dictionary

## `model_input/monthly_model_input.csv`

One row per `point_id`–`Year`–`Month`; 74,100 rows covering 247 sites and January 2001 through December 2025.

| Field | Unit / type | Meaning |
|---|---|---|
| `point_id` | string | Three-digit study-site identifier. |
| `Lon`, `Lat` | decimal degrees | Study-site longitude and latitude (WGS 84). |
| `Year`, `Month` | integer | Target calendar year and month. |
| `daima` | integer | Vegetation-stratum code. |
| `aspect` | degrees | Terrain aspect. |
| `NPP` | kg C m^-2 month^-1 | MOD17A2HGF PsnNet-based monthly reference target used by the modelling workflow; reported model errors are converted to g C m^-2 month^-1. |
| `LST_Day_1km` | degrees C | Monthly mean daytime land-surface temperature. |
| `NDVI` | unitless | Monthly mean MODIS NDVI. |
| `Precipitation_mm` | mm month^-1 | Monthly total precipitation. |
| `VPD` | kPa | Vapour-pressure deficit derived from 2-m temperature and dew point. |
| `surface_solar_radiation_downwards_sum` | MJ m^-2 month^-1 | Monthly downward surface solar radiation. |
| `temperature_2m` | degrees C | Monthly mean 2-m air temperature. |
| `volumetric_soil_water_layer_1` | m3 m^-3 | Monthly mean volumetric soil water in ERA5-Land layer 1. |
| `elevation` | m | Copernicus DEM elevation. |
| `slope` | degrees | Terrain slope. |

`NDVI=-9999` is the recorded sentinel value for `point_id=233` in October 2022.

## `model_input/study_sites_247.csv`

| Field | Unit / type | Meaning |
|---|---|---|
| `point_id` | string | Three-digit site identifier. |
| `Lon`, `Lat` | decimal degrees | Fixed study-site coordinates used by the GEE workflow. |
| `daima` | integer | Vegetation-stratum code. |
| `MOD17_sample_Lon`, `MOD17_sample_Lat` | decimal degrees | MOD17 sampled-pixel centre used in the AppEEARS request. |

## `model_input/unseen_site_fold_assignment.csv`

| Field | Type | Meaning |
|---|---|---|
| `point_id` | string | Three-digit site identifier. |
| `fold_id` | string | Fixed coordinate-clustered holdout fold (`fold_1`–`fold_5`). |

## `mod17_quality/appeears_MOD17A2HGF_point_sample.csv`

This table contains 284,297 composite-level records for 247 sites from 2000-12-26 through 2025-12-27.

| Field group | Fields | Meaning |
|---|---|---|
| Site and date | `Category`, `ID`, `point_id`, `Latitude`, `Longitude`, `composite_start_date` | AppEEARS site identity, sampled-pixel coordinates and MOD17 composite start date. |
| Native pixel identity | `MODIS_Tile`, `MOD17A2HGF_061_Line_Y_500m`, `MOD17A2HGF_061_Sample_X_500m` | MODIS tile and grid position. |
| Product values | `PsnNet`, `Psn_QC` | AppEEARS-scaled PsnNet and raw Psn_QC values. |
| Validity | `PsnNet_valid`, `Psn_QC_valid` | Availability flags for product values. |
| Decoded QA | `MODLAND`, `sensor_bit`, `dead_detector`, `cloud_state`, `SCF_confidence` | Fields decoded from `Psn_QC`. |
| Analysis flags | `psn_good_usable`, `psn_strict_best`, `psn_fill_failed`, `psn_cloud`, `psn_nonclear` | Quality categories used in the MOD17 quality analysis. |

## `annual_reference/MOD17A3HGF_site_year.csv`

One row per site and year; 6,175 rows covering 247 sites and 2001–2025.

| Field group | Fields | Meaning |
|---|---|---|
| Identity | `point_id`, `Lon`, `Lat`, `Year` | Site and year. |
| Annual productivity | `Npp_raw`, `Npp_scaled_kgC_m2_yr`, `Npp_scaled_gC_m2_yr` | Raw and scaled MOD17A3HGF annual Npp. |
| Quality | `Npp_QC_raw`, `npp_missing_or_masked`, `qc_missing_or_masked` | Annual QA and availability flags. |
| Product provenance | `product`, `collection`, `extraction_scale_m`, `product_crs`, `product_nominal_scale_m` | Product and extraction metadata. |
| Source record | `point_source_sha256`, `script_execution_timestamp_utc` | Site-table identity and extraction timestamp. |

## `source_assets/vegetation_roi.gpkg`

Layer `vegetation_roi` contains six WGS 84 multipolygon features.

| Field | Meaning |
|---|---|
| `daima` | Vegetation-stratum code. |
| `vegetation_class` | English vegetation class name. |
