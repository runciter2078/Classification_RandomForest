#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPY Random Forest — Hyperparameter Search
==========================================

Standalone script dedicated exclusively to hyperparameter tuning of a
Random Forest classifier for predicting positive entry days on SPY,
optimizing precision with time-aware cross-validation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rf_hypertune")

FEATURE_COLUMNS = [
    "1", "31", "42", "46", "47", "48", "60", "68", "76", "77",
    "93", "171", "173", "191", "221", "225", "237", "FECHA.month",
]
TARGET_COLUMN = "CLASIFICADOR"
USE_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS
RANDOM_STATE = 15

@dataclass
class SearchConfig:
    data_path: Optional[str] = None
    use_colab_upload: bool = False
    train_ratio: float = 0.75
    n_iter_search: int = 512
    cv_splits: int = 5
    n_estimators: int = 128
    output_dir: Path = Path("output")
    random_state: int = RANDOM_STATE

def load_data(config: SearchConfig) -> pd.DataFrame:
    if config.use_colab_upload:
        try:
            from google.colab import files  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--colab fue especificado pero google.colab no está disponible."
            ) from exc
        uploaded = files.upload()
        filename = next(iter(uploaded))
        logger.info("Archivo subido: %s (%d bytes)", filename, len(uploaded[filename]))
        df = pd.read_csv(io.StringIO(uploaded[filename].decode("utf-8")), sep=",", usecols=USE_COLUMNS)
    else:
        if config.data_path is None:
            raise ValueError("Debes indicar --data-path o usar --colab.")
        path = Path(config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        df = pd.read_csv(path, sep=",", usecols=USE_COLUMNS)
        logger.info("Dataset cargado desde %s: %s filas x %s columnas", path, *df.shape)
    return df

def split_dataset(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(df) * train_ratio)
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    logger.info("Train: %d filas | Test: %d filas", len(train), len(test))
    return train, test

def build_param_distributions() -> dict:
    """
    'auto' fue eliminado de max_features en scikit-learn >= 1.3; se
    sustituye por 'sqrt'/'log2'/None y valores enteros explícitos.
    """
    return {
        "max_features": ["sqrt", "log2", None] + list(range(1, len(FEATURE_COLUMNS) + 1)),
        "max_depth": list(range(2, 21)) + [None],
        "min_samples_split": sp_randint(2, 130),
        "min_samples_leaf": sp_randint(1, 130),
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
        "criterion": ["gini", "entropy"],
        "min_weight_fraction_leaf": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "max_leaf_nodes": list(range(2, 21)) + [None],
    }

def _log_top_results(cv_results: dict, n_top: int = 3) -> None:
    for rank in range(1, n_top + 1):
        candidates = np.flatnonzero(cv_results["rank_test_score"] == rank)
        for candidate in candidates:
            logger.info(
                "Rank %d | score medio: %.3f (std %.3f) | params: %s",
                rank,
                cv_results["mean_test_score"][candidate],
                cv_results["std_test_score"][candidate],
                cv_results["params"][candidate],
            )

def hyperparameter_search(x_train: pd.DataFrame, y_train: pd.Series, config: SearchConfig) -> dict:
    """
    Búsqueda aleatoria optimizando precisión: se asume que una señal de
    entrada errónea (falso positivo) es más costosa que una oportunidad
    no aprovechada. Validación cruzada temporal (TimeSeriesSplit).
    """
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators, n_jobs=-1, random_state=config.random_state
    )
    scorer = make_scorer(precision_score, average="binary", zero_division=0)
    cv = TimeSeriesSplit(n_splits=config.cv_splits)

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=build_param_distributions(),
        n_iter=config.n_iter_search,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
        verbose=1,
    )
    search.fit(x_train, y_train)
    _log_top_results(search.cv_results_, n_top=3)
    logger.info("Mejores hiperparámetros: %s", search.best_params_)
    return search.best_params_

def save_results(best_params: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "best_hyperparameters.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2, default=str)
    logger.info("Hiperparámetros guardados en %s", output_dir / "best_hyperparameters.json")

def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros para el Random Forest de SPY.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--n-iter", type=int, default=512)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    return SearchConfig(
        data_path=args.data_path,
        use_colab_upload=args.colab,
        train_ratio=args.train_ratio,
        n_iter_search=args.n_iter,
        cv_splits=args.cv_splits,
        n_estimators=args.n_estimators,
        output_dir=Path(args.output_dir),
    )

def main() -> None:
    config = parse_args()
    df = load_data(config)
    train, _ = split_dataset(df, config.train_ratio)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    best_params = hyperparameter_search(x_train, y_train, config)
    save_results(best_params, config.output_dir)

if __name__ == "__main__":
    main()
