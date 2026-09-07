# Data sources

| File | Public source |
|---|---|
| `model_input/monthly_model_input.csv` | MODIS MOD17A2HGF, MOD11A2 and MOD13A1; ERA5-Land monthly aggregates; Copernicus DEM, accessed through Google Earth Engine. |
| `model_input/study_sites_247.csv` | Study-site coordinates and MOD17 sampled-pixel centres. |
| `model_input/unseen_site_fold_assignment.csv` | Coordinate-clustered fold assignments for the 247 study sites. |
| `mod17_quality/appeears_MOD17A2HGF_point_sample.csv` | NASA AppEEARS `MOD17A2HGF.061`, layers `PsnNet_500m` and `Psn_QC_500m`. |
| `annual_reference/MOD17A3HGF_site_year.csv` | Google Earth Engine `MODIS/061/MOD17A3HGF`, layers `Npp` and `Npp_QC`. |
| `source_assets/vegetation_roi.gpkg` | Study-region vegetation strata used for site generation. |

Monthly climate variables use `ECMWF/ERA5_LAND/MONTHLY_AGGR`; terrain variables use `COPERNICUS/DEM/GLO30`.
