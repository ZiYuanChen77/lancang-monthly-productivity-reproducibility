# Data acquisition

## Google Earth Engine

- `gee/export_monthly_model_inputs.js`: monthly MODIS, ERA5-Land and Copernicus DEM inputs for the 247 study sites.
- `gee/export_annual_mod17_reference.js`: `MODIS/061/MOD17A3HGF` `Npp` and `Npp_QC`, 2001–2025.
- `gee/export_mod13a1_snow_qa.js`: monthly `MODIS/061/MOD13A1` `DetailedQA` snow/ice flag and composite count used by the cold/snow sensitivity.

Set the required GEE asset identifiers at the top of each script.

## NASA AppEEARS

`appeears/prepare_appeears_request.py` prepares the 247-point request for `MOD17A2HGF.061`, layers `PsnNet_500m` and `Psn_QC_500m`.

```text
python code/data_acquisition/appeears/prepare_appeears_request.py --study-sites ../public_data_v1.1/model_input/study_sites_247.csv --output-dir runtime/appeears_request
```
