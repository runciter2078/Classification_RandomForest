#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Classifier for SPY Data

This script is designed for use in Google Colab. It allows the user to upload a CSV dataset,
performs data loading, hyperparameter tuning using RandomizedSearchCV for a Random Forest classifier,
trains the final model, evaluates it using classification reports and confusion matrices, and displays
feature importances.
"""

import pandas as pd
import numpy as np
from google.colab import files
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
import seaborn as sns


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


def split_dataset(df, train_ratio=0.80):
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


def hyperparameter_search(x_train, y_train, n_iter_search=80):
    """
    Perform hyperparameter tuning using RandomizedSearchCV for a Random Forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=512, n_jobs=-1)
    param_dist = {
        "max_depth": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, None],
        "max_features": sp_randint(1, 18),
        "min_samples_split": sp_randint(2, 95),
        "min_samples_leaf": sp_randint(1, 95),
        "bootstrap": [True, False],
        "class_weight": ['balanced', None],
        "criterion": ["gini", "entropy"]
    }
    random_search = RandomizedSearchCV(clf, scoring='f1', param_distributions=param_dist,
                                       n_iter=n_iter_search, random_state=42)
    random_search.fit(x_train, y_train)
    print("Hyperparameter search results:")
    report(random_search.cv_results_, n_top=3)
    return random_search.best_params_


def train_final_model(x_train, y_train, best_params):
    """
    Train the final Random Forest model with the given best hyperparameters.
    """
    clf_rf = RandomForestClassifier(
        n_estimators=1024,
        n_jobs=-1,
        oob_score=False,
        random_state=42,
        **best_params
    )
    clf_rf.fit(x_train, y_train)
    return clf_rf


def evaluate_model(clf, x_test, y_test):
    """
    Evaluate the model by printing the classification report and confusion matrix.
    """
    preds = clf.predict(x_test)
    print("Random Forest Classification Report:\n")
    print(classification_report(y_true=y_test, y_pred=preds))
    print("Confusion Matrix:")
    cm = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'])
    print(cm)
    return preds


def show_feature_importances(clf, features):
    """
    Display the feature importances of the trained model.
    """
    importance_df = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
    print("Feature Importances:")
    print(importance_df)
    print("\nMaximum Importance:", clf.feature_importances_.max())


def main():
    # Define the columns to use from the CSV file
    usecols = ['CLASIFICADOR', '1', '31', '42', '46', '47', '48', '60',
               '68', '76', '77', '93', '171', '173', '191', '221', '225', '237', 'FECHA.month']

    # Upload dataset file
    filename, file_io = upload_data()
    if filename is None:
        print("No file uploaded.")
        return

    # Load dataset
    df = load_dataset(file_io, usecols)

    # Split dataset into training and testing sets
    train, test = split_dataset(df, train_ratio=0.80)

    # Define features and target
    features = df.columns[1:]
    x_train = train[features]
    y_train = train['CLASIFICADOR']
    x_test = test[features]
    y_test = test['CLASIFICADOR']

    # Hyperparameter search
    best_params = hyperparameter_search(x_train, y_train, n_iter_search=80)
    print("Best parameters:", best_params)

    # Train final Random Forest model using best parameters
    clf_rf = train_final_model(x_train, y_train, best_params)

    # Evaluate the model
    evaluate_model(clf_rf, x_test, y_test)

    # Show feature importances
    show_feature_importances(clf_rf, features)


if __name__ == '__main__':
    main()
