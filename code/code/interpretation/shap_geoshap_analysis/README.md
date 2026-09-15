# SHAP and GeoSHAP

```text
python code/interpretation/shap_geoshap_analysis/run_global_shap.py --list-runs
python code/interpretation/shap_geoshap_analysis/run_global_shap.py --run-id background_BG256_D1
python code/interpretation/shap_geoshap_analysis/run_global_shap.py --run-id seed_2024
python code/interpretation/shap_geoshap_analysis/run_global_shap.py --run-id seed_3407
```

Run each remaining ID shown by `--list-runs` with `--run-id <ID>`, then summarize:

```text
python code/interpretation/shap_geoshap_analysis/summarize_global_shap.py
python code/interpretation/shap_geoshap_analysis/summarize_geoshap.py --phase validate
python code/interpretation/shap_geoshap_analysis/summarize_geoshap.py --phase run
python code/interpretation/shap_geoshap_analysis/summarize_vegetation_temporal_stability.py validate
python code/interpretation/shap_geoshap_analysis/summarize_vegetation_temporal_stability.py run
```

Sample keys: `data_documentation/shap_sample_keys/`.

Inputs: W36 tensors in `data/processed/` and the three trained checkpoints in `runtime/runs/canonical/final_seed_runs/seed_<seed>/`.

Primary global and spatial summaries average per-seed absolute contributions for seeds 42, 2024 and 3407. Background/explained-sample sensitivity comparisons use seed 42 with their specified sample keys; seed stability compares all three models.

Validation files are written under `runtime/runs/*/validation/`; scientific results are written under the corresponding `results/`, `summary/`, and `figures/` directories.
