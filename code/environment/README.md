# Software and hardware environment

`requirements.txt` defines compatible package ranges for running the public workflow.

Recorded study environments:

| Workflow | Python | NumPy | pandas | scikit-learn | PyTorch | Other |
|---|---:|---:|---:|---:|---:|---|
| Canonical W36, RIME and unseen-site models | 3.9.25 | 1.26.0 | 2.3.1 | 1.6.1 | 2.7.1+cu118 | CUDA 11.8; mealpy 3.0.3 |
| Model comparison | 3.12.7 | 1.26.4 | 2.2.2 | 1.5.1 | 2.5.1 | — |
| Grouped bootstrap | 3.12.13 | 2.3.5 | 3.0.1 | — | — | — |
| SHAP analyses | 3.9.25 | 1.26.0 | 2.3.1 | 1.6.1 | 2.7.1+cu118 | SHAP 0.49.1; CPU execution |

Neural-model runs used Windows build 10.0.26200 and an NVIDIA GeForce RTX 3060 Laptop GPU. The modelling code also supports CPU execution.

Recommended setup:

```text
python -m venv .venv
# activate the environment using the command for your operating system
python -m pip install -r environment/requirements.txt
```
