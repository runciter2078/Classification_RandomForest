# SPY Random Forest Classifier

This repository implements a Random Forest classifier in Python to predict positive entry days for the SPY stock index. The project provides two main scripts/notebooks for different stages of the machine learning workflow, as well as an integrated Python script for the full pipeline.

## Repository Structure

- **notebooks/** (if you prefer working interactively)
  - `SPY_Random_Forest_clasificador.ipynb` – Notebook containing the complete pipeline for data loading, splitting, training, evaluation, and feature importance analysis.
  - `Random_Forest_Hypertune.ipynb` – Notebook dedicated to hyperparameter tuning.
- **src/** (scripts)
  - `spy_random_forest_classifier.py` – Script implementing the full pipeline.
  - `random_forest_hypertune.py` – Script focusing on hyperparameter tuning only.
- `README.md` – This file.
- `LICENSE` – The project license.

## Features

- **Interactive Data Upload:**  
  Designed for Google Colab, allowing users to easily upload a CSV dataset.
  
- **Hyperparameter Tuning:**  
  Uses `RandomizedSearchCV` to explore various hyperparameter configurations for the Random Forest classifier.
  
- **Model Evaluation:**  
  Generates classification reports and confusion matrices to assess the model's performance.
  
- **Feature Importance Analysis:**  
  Displays feature importances to help understand the contribution of each variable.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn
- For Google Colab usage: access to `google.colab` for file uploads

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/Classification_RandomForest.git
   ```

2. (Optional) Rename the repository folder to `SPY_RandomForest_Classifier` for clarity.

3. Navigate to the project directory:

   ```bash
   cd Classification_RandomForest
   ```

## Usage

### In Google Colab

1. Open one of the notebooks from the `notebooks/` folder:
   - For hyperparameter tuning, open `Random_Forest_Hypertune.ipynb`.
   - For the complete training and evaluation pipeline, open `SPY_Random_Forest_clasificador.ipynb`.
2. Run the cells sequentially. When prompted, upload your CSV file (e.g., `SPYV3.csv` or `SPYV3-18VAR.csv`).
3. Follow the instructions provided within the notebooks.

### As Python Scripts

#### Full Pipeline

1. Place your CSV file in the project directory.
2. Run the full pipeline script:

   ```bash
   python src/spy_random_forest_classifier.py
   ```

#### Hyperparameter Tuning Only

1. Place your CSV file in the project directory.
2. Run the hyperparameter tuning script:

   ```bash
   python src/random_forest_hypertune.py
   ```

## Notes

- **Hyperparameter Tuning:**  
  The hyperparameter search uses a specified number of iterations (default is 80 for the full pipeline script and 512 for the tuning script). Adjust `n_iter_search` as needed based on your dataset size and available computational resources.

- **Model Evaluation:**  
  The final model is evaluated using a classification report and a confusion matrix to provide detailed performance metrics.

- **Branch Consolidation:**  
  All content has been merged into the main branch to maintain a single coherent codebase.

## License

This project is distributed under the [MIT License](LICENSE).
