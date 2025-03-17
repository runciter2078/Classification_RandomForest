#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Hyperparameter Tuning for SPY Data

This script is designed for use in Google Colab (or any environment that supports file upload).
It allows the user to upload a CSV dataset, loads the dataset, splits it into training and testing sets,
and performs hyperparameter tuning using RandomizedSearchCV for a Random Forest classifier.
The script prints the top models along with their parameters.
"""

import pandas as pd
import numpy as np
from google.colab import files
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import precision_score, make_scorer
import warnings

warnings.filterwarnings('ignore')


def upload_data():
    """
    Upload a CSV file using Google Colab.
    Returns the filename and an io.StringIO object.
    """
    uploaded = files.upload()
    for fn in uploaded.keys():
        print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
        return fn, io.StringIO(uploaded[fn].decode('utf-8'))
    return None, None


def load_dataset(file_io, usecols):
    """
    Load the dataset from the uploaded file using the specified columns.
    """
    df = pd.read_csv(file_io, sep=',', usecols=usecols)
    print("Dataset head:")
    print(df.head())
    return df


def split_dataset(df, train_ratio=0.75):
    """
    Split the dataset into training and testing sets.
    """
    n_train = int(len(df) * train_ratio)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    print("Training samples:", len(train))
    print("Testing samples:", len(test))
    return train, test


def report(results, n_top=3):
    """
    Report the top n models from the hyperparameter search.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank:", i)
            print("Mean validation score: {:.3f} (std: {:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters:", results['params'][candidate])
            print("")


def hyperparameter_search(x_train, y_train, n_iter_search=512):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for a Random Forest classifier.
    """
    clf = RandomForestClassifier(n_jobs=-1)
    param_grid = { 
        'n_estimators': [128],
        'max_features': ['auto', 'sqrt', 'log2', 7, 6, 5, 4, 3, 2, 1, None],
        "max_depth": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, None],
        "min_samples_split": sp_randint(2, 130),
        "min_samples_leaf": sp_randint(1, 130),
        'bootstrap': [True, False],
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy'],
        'n_jobs': [-1],
        "random_state": [15],
        "min_weight_fraction_leaf": [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "max_leaf_nodes": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, None]
    }
    metrica = make_scorer(precision_score, greater_is_better=True, average="binary")
    random_search = RandomizedSearchCV(clf, scoring=metrica, param_distributions=param_grid,
                                       n_iter=n_iter_search, random_state=15)
    random_search.fit(x_train, y_train)
    print("Hyperparameter search results:")
    report(random_search.cv_results_, n_top=3)
    return random_search.best_params_


def main():
    # Define the columns to use from the CSV file
    usecols = ['CLASIFICADOR', '1', '31', '42', '46', '47', '48', '60', '68', '76', '77', 
               '93', '171', '173', '191', '221', '225', '237', 'FECHA.month']
    
    # Upload dataset file
    filename, file_io = upload_data()
    if filename is None:
        print("No file uploaded.")
        return
    
    # Load dataset
    df = load_dataset(file_io, usecols)
    
    # Split dataset into training and testing sets
    train, test = split_dataset(df, train_ratio=0.75)
    
    # Define features and target (using training data for hyperparameter search)
    features = df.columns[1:]
    x_train = train[features]
    y_train = train['CLASIFICADOR']
    
    # Perform hyperparameter search
    best_params = hyperparameter_search(x_train, y_train, n_iter_search=512)
    print("Best parameters found:")
    print(best_params)
    
    
if __name__ == '__main__':
    main()
