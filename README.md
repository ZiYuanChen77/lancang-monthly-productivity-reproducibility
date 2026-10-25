# Lancang monthly productivity

Reproducibility code for SRS-D-26-00418, revision R2.

Inputs: `public_data/`. Code rebuilds NDVI exclusion keys and analysis inputs. Field definitions and sources are in `public_data/documentation/`.

Dependencies: `code/environment/requirements.txt`; a CUDA-enabled GPU for the recorded training environment; R 4.6.1 with mgcv 1.9-4. Set `RSCRIPT` if Rscript is not on PATH.

From the repository root:

```text
python code/run_revision.py prepare
python code/run_revision.py clean
python code/run_revision.py shap
python code/run_revision.py shift
python code/run_revision.py spatial
python code/run_revision.py water
```

The canonical workflow (`code/code/canonical_pipeline.py`, `code/run_pipeline.py`) builds predictions and SHAP caches. `clean`, `shap` and `shift` summarize these outputs on the NDVI-clean domain. Copy `code/configs/config.example.yaml` to `config.yaml` in the same folder for model comparison.

Seeds: 42/2024/3407. W36 uses 11,820 evaluation records per seed; spatial comparisons use 11,818. The primary SHAP summary retains 997 of the 1,000 E0 sample keys. Use `clean --evaluation-keys` for the common domain and `--arm`/`--paired-arm` for spatial arms.

The spatial entry point runs fold-specific window and RIME selection; fixed-W36 refits provide the reference arm.

Water-energy comparisons use 2,000 joint site bootstrap fits. Cold/snow sensitivity inputs use the MOD13A1 QA export through `prepare --ndvi-qa`. Table S33 reverses the sensitivity scripts' radiation contrast.

Outputs stay under `code/data/`, `code/runtime/` and `code/outputs/`. Further options: `--help`.
