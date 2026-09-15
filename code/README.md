# W36 monthly productivity modelling

This repository contains data-acquisition, preprocessing, modelling, evaluation, uncertainty, attribution, and water–energy analysis code for the accompanying study. It uses `public_data_v1.1`.

## Setup

```text
python -m venv .venv
python -m pip install -r environment/requirements.txt
```

Copy `configs/config.example.yaml` to `configs/config.yaml` and set local paths. A typical layout is:

```text
repository/
  reproducibility_code/
  public_data_v1.1/
```

## Prepare W36 model inputs

```text
python code/preprocessing/build_w36_inputs.py --input ../public_data_v1.1/model_input/monthly_model_input.csv --output-dir data/processed
```

This creates aligned W6, W12, W18, W24, W36 and W48 tensors for Train (2005–2018), Validation (2019–2021) and Evaluation (2022–2025).

## Canonical model

```text
python code/canonical_pipeline.py select-window --data-dir data/processed --window-seed-summary data_documentation/window_validation_by_seed.csv --out-dir runtime/runs/canonical
python code/canonical_pipeline.py rime-search --data-dir data/processed --window-selection runtime/runs/canonical/window_selection_manifest.json --out-dir runtime/runs/canonical
python code/canonical_pipeline.py train-seeds --data-dir data/processed --best-hp runtime/runs/canonical/rime_search/best_hyperparameters.json --rime-manifest runtime/runs/canonical/rime_search/rime_search_manifest.json --out-dir runtime/runs/canonical
python code/canonical_pipeline.py lock-seeds --data-dir data/processed --seed-summary runtime/runs/canonical/final_seed_runs/seed_validation_summary.csv --seed-training-manifest runtime/runs/canonical/final_seed_runs/seed_training_manifest.json --out-dir runtime/runs/canonical
```

The fixed training seeds are 42, 2024 and 3407. Stochastic-model summaries use the three-seed mean and sample standard deviation (`ddof=1`).

## Comparison models and unseen-site evaluation

```text
python run_pipeline.py help-workflow
python run_pipeline.py validate-workflow
python run_pipeline.py tune-models
python run_pipeline.py train-models
python run_pipeline.py select-baselines
python code/canonical_pipeline.py evaluate --data-dir data/processed --lock-manifest runtime/runs/canonical/pre_evaluation_lock_manifest.json --out-dir runtime/runs/canonical --confirm-evaluation I_ACCEPT_FINAL_EVALUATION
python run_pipeline.py evaluate-models
python run_pipeline.py evaluate-baselines
python run_pipeline.py metrics
python run_pipeline.py validate-structure
python run_pipeline.py workbook
python run_pipeline.py fit-unseen-site --site-table ../public_data_v1.1/model_input/study_sites_247.csv
python run_pipeline.py evaluate-unseen-site --site-table ../public_data_v1.1/model_input/study_sites_247.csv
```

## Water–energy GAMM inputs

Generate GAMM inputs in `data/water_energy/`:

```text
python run_pipeline.py prepare-water-energy --public-data-root ../public_data_v1.1
```

The public AppEEARS table supplies the MOD17 QA and day-weighted target fields. The cold/snow sensitivity uses the optional MOD13A1 monthly QA export produced by `code/data_acquisition/gee/export_mod13a1_snow_qa.js`.

Module descriptions are in `code/README.md`. SHAP/GeoSHAP commands are in `code/interpretation/shap_geoshap_analysis/README.md`, and GAMM details are in `code/water_energy_analysis/README.md`.


## Public release scope

The repository uses the published point-to-fold assignment for unseen-site evaluation. Comparison-model selection uses Train/Validation data, and Evaluation is opened only after model selection is fixed.
