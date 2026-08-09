# CMPS 3500 Final

Coursework exploring data cleaning, feature preparation, and neural-network modeling for Los Angeles crime-status data.

[![Python](https://img.shields.io/badge/python-analysis-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

## Overview

The repository contains a final Python pipeline, Jupyter notebooks, an exploratory analysis notebook, and a separate `bolt-job/` model runner. The project uses Los Angeles crime records to prepare features and predict crime status categories.

## Contents

- `Final/final.py` — interactive final pipeline with data loading, cleaning, modeling, and prediction helpers
- `Notebooks/` — starter, cleaning, and model notebooks plus an exported cleaning HTML file
- `final/` — committed training and test CSV files and exploratory analysis
- `bolt-job/` — separate runner, setup script, configuration, and saved PyTorch model

## Methodology

The final script loads CSV data, removes or transforms malformed fields, engineers time and categorical features, and defines PyTorch model classes including a tabular Transformer and MLP. It also imports pandas, NumPy, scikit-learn, matplotlib, seaborn, tqdm, imbalanced-learn, and optional PyTorch functionality.

## Usage

The repository does not include a dependency manifest or a verified one-command runner. Review the input paths and interactive entry point in `Final/final.py` before execution. The committed datasets are under `final/` and should not be confused with the external course data paths described in the historical README.

## Status and limitations

This is course project material. Model metrics and sample output in earlier documentation are not treated as independently reproducible claims here. The data files may contain sensitive or restricted source data; use them only according to their provenance and course permissions.
