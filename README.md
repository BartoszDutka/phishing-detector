# Phishing URL Detector

A machine learning project for phishing URL classification using the PhiUSIIL dataset.

## Features

- Dataset loading and preprocessing
- Random Forest and Logistic Regression model evaluation
- cross-validation and feature selection
- ROC and feature importance analysis
- noise sensitivity experiments

## Technology stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, KaggleHub, Jupyter Notebook.

## Getting started

Install dependencies with `python -m pip install -r requirements.txt`, install Jupyter, and open `main.ipynb`. Dataset configuration is in `config.py`. Review dataset access and label definitions before running experiments. Model scores should be interpreted with respect to data splits and potential leakage; this is not a production phishing protection service.
