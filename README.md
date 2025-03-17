# SPY Random Forest Classifier

This repository implements a Random Forest classifier in Python to predict positive entry days for the SPY stock index. The project provides two main Python scripts for different stages of the machine learning workflow.

## Repository Structure

- **src/**
  - `spy_random_forest_classifier.py` – Script implementing the full pipeline for data loading, splitting, hyperparameter tuning, model training, evaluation, and feature importance analysis.
  - `random_forest_hypertune.py` – Script focusing solely on hyperparameter tuning.
- `README.md` – This file.
- `LICENSE` – The project license.

## Features

- **Interactive Data Upload:**  
  Designed for use in Google Colab, allowing users to easily upload a CSV dataset.
  
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

### Full Pipeline Script

1. Place your CSV file in the project directory.
2. Run the full pipeline script:

   ```bash
   python src/spy_random_forest_classifier.py
   ```

### Hyperparameter Tuning Script

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
  All content has been merged into the main branch to maintain a single, coherent codebase.

## License

This project is distributed under the [MIT License](LICENSE).
