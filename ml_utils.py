"""Shared utilities for training and the Streamlit evaluation application.

The target encoding is 0 = benign and 1 = malignant. This makes the
positive class (1) the clinically important malignant class, consistently
for every reported binary metric.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / 'data' / 'uci_wdbc_dataset.csv'
TEST_DATA_PATH = PROJECT_ROOT / 'test_data.csv'
MODEL_DIR = PROJECT_ROOT / 'model'
RESULTS_DIR = PROJECT_ROOT / 'results'
TARGET_COLUMN = 'diagnosis'
POSITIVE_CLASS = 1
CLASS_NAMES = ['Benign (0)', 'Malignant (1)']


def read_feature_names() -> list[str]:
    """Read the recorded feature contract; avoids drift between app and training."""
    metadata_path = RESULTS_DIR / 'experiment.json'
    with metadata_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)['feature_names']


def load_labeled_csv(source, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Check uploaded CSV and return numeric predictors plus validated 0/1 target.

    Supports the project's diagnosis labels B/M. Other unexpected target values
    are rejected rather than silently coerced to a different meaning.
    """
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError('CSV has no observations.')
    if data.columns.duplicated().any():
        raise ValueError('CSV has duplicate column names.')
    missing_columns = [name for name in [*feature_names, TARGET_COLUMN] if name not in data]
    if missing_columns:
        raise ValueError('Required columns are missing: ' + ', '.join(missing_columns))

    labels = data[TARGET_COLUMN].astype('string').str.strip().str.upper()
    if not labels.isin(['B', 'M']).all():
        raise ValueError("The diagnosis target must contain only 'B' (benign) or 'M' (malignant).")
    target = labels.map({'B': 0, 'M': 1}).astype(int)

    raw_features = data.loc[:, feature_names]
    numeric_features = raw_features.apply(pd.to_numeric, errors='coerce')
    bad_cells = numeric_features.isna() & raw_features.notna()
    if bad_cells.any().any():
        affected = list(bad_cells.columns[bad_cells.any()])
        raise ValueError('Non-numeric values in feature columns: ' + ', '.join(affected))
    if numeric_features.isna().any().any():
        raise ValueError('Missing feature values found. This assignment uses a complete dataset.')
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise ValueError('Infinite feature values are not allowed.')
    return numeric_features, target


def evaluate_classifier(model, features: pd.DataFrame, target: pd.Series) -> dict:
    """Calculate all six required evaluation metrics without fitting the model."""
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    malignant_col = classes.index(POSITIVE_CLASS)
    malignant_scores = probabilities[:, malignant_col]

    # AUC is undefined if the uploaded data contains only one target class.
    auc = float(roc_auc_score(target, malignant_scores)) if target.nunique() == 2 else None
    return {
        'Accuracy': float(accuracy_score(target, predicted)),
        'AUC': auc,
        'Precision': float(precision_score(target, predicted, pos_label=1, zero_division=0)),
        'Recall': float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        'F1': float(f1_score(target, predicted, pos_label=1, zero_division=0)),
        'MCC': float(matthews_corrcoef(target, predicted)),
    }


def detailed_evaluation(model, features: pd.DataFrame, target: pd.Series) -> dict:
    predicted = model.predict(features)
    scores = evaluate_classifier(model, features, target)
    cm = confusion_matrix(target, predicted, labels=[0, 1])
    report = classification_report(
        target, predicted, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return {'metrics': scores, 'matrix': cm.tolist(), 'report': report}
