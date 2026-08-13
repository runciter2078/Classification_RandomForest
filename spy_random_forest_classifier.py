#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPY Random Forest Classifier
=============================

Full pipeline for predicting positive entry days on the SPY ETF: data
loading, chronological train/test split, time-aware hyperparameter
search, final model training, evaluation and feature importance
analysis. Can be run locally or inside Google Colab.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # backend seguro fuera de un entorno interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spy_random_forest")

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

FEATURE_COLUMNS = [
    "1", "31", "42", "46", "47", "48", "60", "68", "76", "77",
    "93", "171", "173", "191", "221", "225", "237", "FECHA.month",
]
TARGET_COLUMN = "CLASIFICADOR"
USE_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS
RANDOM_STATE = 42

@dataclass
class PipelineConfig:
    data_path: Optional[str] = None
    use_colab_upload: bool = False
    train_ratio: float = 0.80
    n_iter_search: int = 80
    cv_splits: int = 5
    n_estimators_final: int = 1024
    output_dir: Path = Path("output")
    random_state: int = RANDOM_STATE

# ----------------------------------------------------------------------
# Carga y validación de datos
# ----------------------------------------------------------------------

def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Carga el dataset desde Colab (upload interactivo) o desde disco."""
    if config.use_colab_upload:
        try:
            from google.colab import files  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--colab fue especificado pero google.colab no está disponible. "
                "Ejecuta este script dentro de un notebook de Google Colab."
            ) from exc
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No se subió ningún archivo.")
        filename = next(iter(uploaded))
        logger.info("Archivo subido: %s (%d bytes)", filename, len(uploaded[filename]))
        buffer = io.StringIO(uploaded[filename].decode("utf-8"))
        df = pd.read_csv(buffer, sep=",", usecols=USE_COLUMNS)
    else:
        if config.data_path is None:
            raise ValueError("Debes indicar --data-path o usar --colab.")
        path = Path(config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        df = pd.read_csv(path, sep=",", usecols=USE_COLUMNS)
        logger.info("Dataset cargado desde %s: %s filas x %s columnas", path, *df.shape)

    _validate_dataset(df)
    return df

def _validate_dataset(df: pd.DataFrame) -> None:
    """Comprobaciones básicas de calidad sobre el dataset cargado."""
    missing_cols = set(USE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing_cols}")

    n_missing = df[USE_COLUMNS].isna().sum().sum()
    if n_missing > 0:
        logger.warning(
            "El dataset contiene %d valores nulos. Este pipeline no los imputa; "
            "trátalos antes de continuar si es necesario.",
            n_missing,
        )

    class_counts = df[TARGET_COLUMN].value_counts(normalize=True)
    logger.info("Distribución de clases:\n%s", class_counts.to_string())
    if class_counts.min() < 0.10:
        logger.warning(
            "Fuerte desbalanceo de clases detectado (clase minoritaria: %.1f%%). "
            "class_weight está incluido en la búsqueda de hiperparámetros.",
            class_counts.min() * 100,
        )

# ----------------------------------------------------------------------
# Split cronológico
# ----------------------------------------------------------------------

def split_dataset(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split cronológico, sin shuffle: el conjunto de test contiene siempre
    las observaciones más recientes, evitando fuga de información temporal.
    """
    n_train = int(len(df) * train_ratio)
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    logger.info("Train: %d filas | Test: %d filas (ratio=%.2f)", len(train), len(test), train_ratio)
    return train, test

# ----------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ----------------------------------------------------------------------

def build_param_distributions() -> dict:
    """
    Espacio de búsqueda de hiperparámetros. 'auto' fue eliminado de
    max_features en scikit-learn >= 1.3; se sustituye por 'sqrt'/'log2'/None
    y valores enteros explícitos.
    """
    return {
        "max_depth": [None, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
        "max_features": ["sqrt", "log2", None] + list(range(1, len(FEATURE_COLUMNS))),
        "min_samples_split": sp_randint(2, 95),
        "min_samples_leaf": sp_randint(1, 95),
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
        "criterion": ["gini", "entropy"],
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

def hyperparameter_search(
    x_train: pd.DataFrame, y_train: pd.Series, config: PipelineConfig
) -> dict:
    """
    Búsqueda aleatoria con validación cruzada temporal (TimeSeriesSplit),
    imprescindible en series financieras para no filtrar información
    futura al conjunto de validación.
    """
    base_clf = RandomForestClassifier(
        n_estimators=512, n_jobs=-1, random_state=config.random_state
    )
    cv = TimeSeriesSplit(n_splits=config.cv_splits)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=build_param_distributions(),
        n_iter=config.n_iter_search,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    _log_top_results(search.cv_results_, n_top=3)
    logger.info("Mejores hiperparámetros: %s", search.best_params_)
    return search.best_params_

# ----------------------------------------------------------------------
# Entrenamiento final
# ----------------------------------------------------------------------

def train_final_model(
    x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, config: PipelineConfig
) -> RandomForestClassifier:
    # oob_score solo es válido si bootstrap=True; best_params puede
    # devolver bootstrap=False, así que se activa condicionalmente.
    oob_score = bool(best_params.get("bootstrap", True))
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators_final,
        n_jobs=-1,
        random_state=config.random_state,
        oob_score=oob_score,
        **best_params,
    )
    clf.fit(x_train, y_train)
    if oob_score:
        logger.info("OOB score del modelo final: %.4f", clf.oob_score_)
    else:
        logger.info("bootstrap=False en los mejores parámetros; OOB score no disponible.")
    return clf

# ----------------------------------------------------------------------
# Evaluación
# ----------------------------------------------------------------------

def evaluate_model(
    clf: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path
) -> pd.DataFrame:
    preds = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]

    report_txt = classification_report(y_test, preds)
    logger.info("Classification report:\n%s", report_txt)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    try:
        auc = roc_auc_score(y_test, proba)
        logger.info("ROC-AUC: %.4f", auc)
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
        ax.set_title(f"Curva ROC (AUC = {auc:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close(fig)
    except ValueError:
        logger.warning("ROC-AUC no calculable (probablemente una sola clase en test).")

    return pd.DataFrame({"y_true": y_test.values, "y_pred": preds, "proba_positive": proba})

def analyze_feature_importance(
    clf: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    """
    Combina la importancia basada en impureza (rápida pero sesgada hacia
    variables de alta cardinalidad) con la importancia por permutación
    (más fiable, calculada sobre el conjunto de test).
    """
    impurity_imp = pd.Series(clf.feature_importances_, index=features, name="impurity")

    perm_result = permutation_importance(
        clf, x_test, y_test, n_repeats=30, random_state=random_state, n_jobs=-1
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=features, name="permutation")

    importance_df = pd.concat([impurity_imp, perm_imp], axis=1).sort_values(
        "permutation", ascending=False
    )
    logger.info("Importancia de variables:\n%s", importance_df.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    importance_df["permutation"].sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_xlabel("Importancia por permutación")
    ax.set_title("Importancia de variables (test set)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    return importance_df

def save_artifacts(clf: RandomForestClassifier, best_params: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "random_forest_model.joblib")
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2, default=str)
    logger.info("Modelo y parámetros guardados en %s", output_dir)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Pipeline completo de Random Forest para predecir días de entrada positivos en SPY."
    )
    parser.add_argument("--data-path", type=str, default=None, help="Ruta al CSV local con los datos.")
    parser.add_argument("--colab", action="store_true", help="Subir el CSV mediante google.colab.files.upload().")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    return PipelineConfig(
        data_path=args.data_path,
        use_colab_upload=args.colab,
        train_ratio=args.train_ratio,
        n_iter_search=args.n_iter,
        cv_splits=args.cv_splits,
        n_estimators_final=args.n_estimators,
        output_dir=Path(args.output_dir),
    )

def main() -> None:
    config = parse_args()

    df = load_data(config)
    train, test = split_dataset(df, config.train_ratio)

    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    best_params = hyperparameter_search(x_train, y_train, config)
    clf = train_final_model(x_train, y_train, best_params, config)

    evaluate_model(clf, x_test, y_test, config.output_dir)
    analyze_feature_importance(
        clf, x_test, y_test, FEATURE_COLUMNS, config.output_dir, config.random_state
    )
    save_artifacts(clf, best_params, config.output_dir)

if __name__ == "__main__":
    main()
