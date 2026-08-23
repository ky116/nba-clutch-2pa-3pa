# Environment setup

This repository does not require committing local Python or R library folders.
The ignored/generated directories `.venv/` and `.r_libs_4.5/` can be recreated
from the requirement files below.

## Python

Recommended Python version: 3.11. Python 3.12.3 was used in the cleanup
environment, but Python 3.11 is the safer default for package-wheel
compatibility across the boosting and Parquet dependencies.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Python requirements include the tree/boosting learners used in the
manuscript pipeline (`xgboost`, `lightgbm`, and `catboost`) and `pyarrow` for
Parquet I/O.

## R

```sh
mkdir -p .r_libs_4.5
Rscript -e 'options(repos = c(CRAN = "https://cloud.r-project.org")); install.packages(readLines("requirements-r.txt"), lib = ".r_libs_4.5")'
export R_LIBS_USER="$PWD/.r_libs_4.5"
```

For a persistent shell setup, add the `R_LIBS_USER` export to your shell profile
or pass it when running the R workflow scripts.

## Quick checks

```sh
. .venv/bin/activate
python -c "import numpy, pandas, pyarrow, sklearn, statsmodels, xgboost, lightgbm, catboost"
R_LIBS_USER="$PWD/.r_libs_4.5" Rscript -e 'library(data.table); library(mgcv)'
```
