# Water–energy GAMM analysis

This module reproduces the 2022–2025 water–energy association analysis. The response is monthly productivity anomaly relative to the 2005–2018 same-site, same-calendar-month baseline, and precipitation/radiation exposures are expressed as historical relative ranks.

## Prepare inputs from the public data package

Generate GAMM inputs from `public_data_v1.1`:

From the repository root:

```text
python run_pipeline.py prepare-water-energy --public-data-root ../public_data_v1.1
```

This uses:

- `model_input/monthly_model_input.csv`
- `mod17_quality/appeears_MOD17A2HGF_point_sample.csv`

and creates `data/water_energy/` with the primary analysis table, contrast-support table, baseline tables, and a local copy of the monthly source table required by the raw-exposure sensitivity.

Expected preparation checks are: primary n = 6,992 across 247 sites; AR sections = 988; A support = 401 records / 215 sites / 4 years; B support = 339 / 182 / 4; MOD17 QA1 n = 5,280; QA2 n = 3,358; day-weighted n = 6,992.

The cold/snow sensitivity additionally uses monthly MOD13A1 `DetailedQA` snow/ice information. Generate it with `code/data_acquisition/gee/export_mod13a1_snow_qa.js`, then run:

```text
python run_pipeline.py prepare-water-energy --public-data-root ../public_data_v1.1 --ndvi-qa path/to/ndvi_monthly_qa_2001_2025.csv
```

## Primary model

The primary Gaussian GAMM uses `mgcv::bam` with an identity link, a tensor-product smooth of precipitation rank and solar-radiation rank (`k = c(5,5)`), four one-dimensional adjustment smooths (`k = 5`), calendar month, year, a `point_id` random intercept, and AR(1) residual dependence.

The fixed contrast compares A = (precipitation rank 0.75, radiation rank 0.25) with B = (0.75, 0.50) and reports Prediction(A) − Prediction(B). The configuration uses seed `20260803`, fixed rho `0.036623315093487`, and a point/site-cluster bootstrap with 2,000 successful replicates. The primary estimate is `-2.94258224191719 g C m^-2 month^-1`, with percentile 95% CI `[-3.559086, -2.211227]`.

Sensitivity analyses use 1,000 successful site-cluster replicates with seed `20260803` and cover adjustment-set robustness, cold/snow exclusion, MOD17 quality filters, day-weighted target construction, full-year and April–September domains, historical-median response definition, and raw-unit exposure definition.

## Environment and run order

Environment: R 4.6.1 and `mgcv` 1.9-4. From an R session at the repository root:

```r
Sys.setenv(WATER_ENERGY_PROJECT_ROOT = normalizePath("."),
           WATER_ENERGY_CODE_DIR = normalizePath("code/water_energy_analysis"))
source(file.path(Sys.getenv("WATER_ENERGY_CODE_DIR"), "validate_primary_inputs.R"))
```

Run the following scripts in order using the same `source()` command. Inputs: `data/water_energy/`. Outputs: `outputs/water_energy/`.

1. Primary: `validate_primary_inputs.R`, `fit_primary_gamm.R`, `validate_primary_residuals.R`, `bootstrap_primary_contrast.R`, `leave_one_year_out.R`.
2. Adjustment robustness: `prepare_adjustment_inputs.R`, `fit_adjustment_sensitivity.R`, `bootstrap_adjustment_sensitivity.R`.
3. Cold/snow: `prepare_cold_snow_inputs.R`, `validate_cold_snow_inputs.R`, `fit_cold_snow_sensitivity.R`, `bootstrap_cold_snow_sensitivity.R`.
4. MOD17 QA/day-weighted: `validate_qa_dayweight_inputs.R`, `fit_qa_dayweight_sensitivity.R`, `bootstrap_qa_dayweight_sensitivity.R`.
5. Domain: `validate_domain_inputs.R`, `fit_domain_sensitivity.R`, `bootstrap_domain_sensitivity.R`.
6. Definition: `prepare_definition_inputs.R`, `validate_definition_inputs.R`, `fit_definition_sensitivity.R`, `bootstrap_definition_sensitivity.R`.
