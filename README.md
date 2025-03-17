# SPY Random Forest Classifier

This project implements a Random Forest classifier in Python to predict positive entry days of the SPY stock index. The repository includes a Google Colab notebook for interactive data analysis, hyperparameter tuning using RandomizedSearchCV, model training, evaluation, and feature importance analysis.

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
- For the Colab version: access to `google.colab` for file uploads

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/Classification_RandomForest.git
   ```

2. *(Optional)* Rename the repository folder to `SPY_RandomForest_Classifier` for clarity.

3. Navigate to the project directory:

   ```bash
   cd Classification_RandomForest
   ```

## Usage

### In Google Colab

1. Open the notebook `SPY_Random_Forest_clasificador.ipynb` in Google Colab.
2. Run the cells sequentially. When prompted, upload your CSV file (e.g., `SPYV3-18VAR.csv`).
3. The notebook will perform data loading, hyperparameter tuning, training, evaluation, and feature importance analysis.

### As a Python Script

An improved version of the code is provided in `spy_random_forest_classifier.py`. To run it:

1. Place your CSV file in the project directory.
2. Execute the script:

   ```bash
   python spy_random_forest_classifier.py
   ```

## Notes

- **Hyperparameter Tuning:**  
  The hyperparameter search uses a specified number of iterations (default is 80). Adjust `n_iter_search` as needed based on your dataset size and available computational resources.

- **Model Evaluation:**  
  The final model is evaluated using a classification report and a confusion matrix to provide detailed performance metrics.

## License

This project is distributed under the [MIT License](LICENSE).
