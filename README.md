# SPY Random Forest Classifier

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.3-orange)
![pandas](https://img.shields.io/badge/pandas-%E2%9C%93-150458)
![numpy](https://img.shields.io/badge/numpy-%E2%9C%93-013243)
![License](https://img.shields.io/badge/license-GPL--3.0-green)

A Random Forest pipeline for predicting positive entry days on the SPY ETF from a set of engineered numerical features. The project follows leakage-safe practices for financial time series: chronological train/test splitting and time-aware cross-validation during hyperparameter search.

> Disclaimer: this project is for educational and research purposes only. It does not constitute financial or investment advice, and past predictive performance does not guarantee future results.

## Repository structure

- `spy_random_forest_classifier.py` — Full pipeline: data loading, chronological train/test split, hyperparameter search (RandomizedSearchCV + TimeSeriesSplit), final model training, evaluation (classification report, confusion matrix, ROC-AUC) and feature importance analysis (impurity-based and permutation-based).
- `random_forest_hypertune.py` — Standalone hyperparameter search script, optimized for precision (minimizing false positive entry signals) over a wider search space.
- `README.md` — This file.
- `LICENSE` — Project license (GPL-3.0).

## Methodology

- Chronological split (no shuffling): the test set always contains the most recent observations, avoiding look-ahead bias typical of financial datasets.
- TimeSeriesSplit cross-validation during hyperparameter search, instead of standard K-Fold, to keep every validation fold strictly posterior to its training fold.
- Precision as the tuning objective in `random_forest_hypertune.py`: a false positive entry signal is assumed to be more costly than a missed opportunity; change the scorer if your use case differs.
- Two complementary feature importance methods in the full pipeline: impurity-based (fast, but biased towards high-cardinality features) and permutation-based (slower, computed on held-out data, more reliable).
- `class_weight` is included in the hyperparameter search space to handle potential class imbalance in the target variable.

## Requirements

- Python 3.9+

```text
pandas
numpy
scikit-learn>=1.3
scipy
matplotlib
seaborn
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

`google.colab` is only required if you run the scripts with the `--colab` flag inside a Google Colab notebook.

## Data format

Both scripts expect a CSV file with a binary target column named `CLASIFICADOR` and the following feature columns:

```text
1, 31, 42, 46, 47, 48, 60, 68, 76, 77, 93, 171, 173, 191, 221, 225, 237, FECHA.month
```

Column names are kept as in the original dataset (anonymized/engineered technical indicators plus a calendar feature). Adjust `FEATURE_COLUMNS` and `TARGET_COLUMN` at the top of each script if you use a different dataset.

## Usage

### Full pipeline

Local execution:

```bash
python spy_random_forest_classifier.py --data-path path/to/data.csv
```

Inside Google Colab, with interactive upload:

```bash
python spy_random_forest_classifier.py --colab
```

Optional arguments:

```text
--train-ratio     Proportion of data used for training (default: 0.80)
--n-iter          RandomizedSearchCV iterations (default: 80)
--cv-splits       Number of TimeSeriesSplit folds (default: 5)
--n-estimators    Trees in the final model (default: 1024)
--output-dir      Output directory for artifacts (default: output)
```

Generated artifacts (under `output/` by default):

```text
random_forest_model.joblib     Trained model, ready to reload with joblib.load
best_params.json               Best hyperparameters found
classification_report.txt      Precision / recall / F1 per class
confusion_matrix.png           Confusion matrix heatmap
roc_curve.png                  ROC curve with AUC
feature_importance.png         Permutation importance plot
```

### Hyperparameter search only

```bash
python random_forest_hypertune.py --data-path path/to/data.csv
```

The same `--colab`, `--train-ratio`, `--n-iter`, `--cv-splits` and `--output-dir` arguments are available; results are saved to `output/best_hyperparameters.json`.

## Notes and limitations

- Missing values are not imputed automatically; the pipeline logs a warning if any are found in the selected columns, and handling them beforehand is the user's responsibility.
- The dataset is not included in this repository.
- `RandomizedSearchCV` results are stochastic; increase `--n-iter` and keep `random_state` fixed for more stable and reproducible results.
## License

Distributed under the [GNU General Public License v3.0](LICENSE).
