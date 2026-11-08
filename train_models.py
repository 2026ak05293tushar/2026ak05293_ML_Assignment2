"""Reproducibly train and evaluate five classifiers on the UCI WDBC dataset.

Run: python train_models.py

Produces five saved, *fitted* pipelines; test_data.csv; evaluation tables; and
experiment metadata. The Streamlit app only LOADS these fitted pipelines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef

from ml_utils import (
    DATASET_PATH, MODEL_DIR, RESULTS_DIR, TEST_DATA_PATH,
    TARGET_COLUMN, evaluate_classifier, detailed_evaluation,
)

RANDOM_SEED = 42
TEST_FRACTION = 0.20
CV_FOLDS = 5
MODEL_SPECIFICATIONS = {
    'Logistic Regression': ('logistic_regression.joblib', True, LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
    'Decision Tree': ('decision_tree.joblib', False, DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=RANDOM_SEED)),
    'KNN': ('knn.joblib', True, KNeighborsClassifier(n_neighbors=7, weights='distance')),
    'Gaussian Naive Bayes': ('gaussian_nb.joblib', False, GaussianNB()),
    'Support Vector Machine': ('svm.joblib', True, CalibratedClassifierCV(
        estimator=SVC(C=1.0, kernel='rbf', random_state=RANDOM_SEED),
        method='sigmoid', cv=CV_FOLDS, ensemble=False,
    )),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_pipeline(needs_scaling: bool, estimator) -> Pipeline:
    """Fit imputation/scaling only within training or cross-validation folds."""
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if needs_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', clone(estimator)))
    return Pipeline(steps)


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f'Dataset missing: {DATASET_PATH}')
    original = pd.read_csv(DATASET_PATH)
    if original.shape != (569, 31):
        raise ValueError(f'Expected 569 observations and 30 predictors + target; got {original.shape}')
    if original.columns.duplicated().any() or original.isna().any().any():
        raise ValueError('Dataset has duplicate column names or missing values.')
    if TARGET_COLUMN not in original or set(original[TARGET_COLUMN].unique()) != {'B', 'M'}:
        raise ValueError('Expected diagnosis labels B and M.')
    predictors = original.drop(columns=[TARGET_COLUMN]); feature_names = predictors.columns.tolist()
    if len(feature_names) != 30 or not all(pd.api.types.is_numeric_dtype(predictors[c]) for c in feature_names):
        raise ValueError('Expected exactly 30 numeric UCI WDBC predictor features.')
    labels = original[TARGET_COLUMN].map({'B': 0, 'M': 1}).astype(int)

    # Stratification preserves the original malignant/benign proportions.
    x_train, x_test, y_train, y_test = train_test_split(
        predictors, labels, test_size=TEST_FRACTION, random_state=RANDOM_SEED,
        stratify=labels,
    )
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.map({0: 'B', 1: 'M'})
    test_frame.to_csv(TEST_DATA_PATH, index=False, float_format='%.12g')
    # Separate training-only CSV is for verification/reproduction; not used by the app.
    train_frame = x_train.copy()
    train_frame[TARGET_COLUMN] = y_train.map({0: 'B', 1: 'M'})
    train_frame.to_csv(RESULTS_DIR / 'training_partition.csv', index=False, float_format='%.12g')

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    summary_rows = []
    analyses = {}
    for name, (artifact_name, scaled, classifier) in MODEL_SPECIFICATIONS.items():
        pipeline = create_pipeline(scaled, classifier)
        # Only training observations are used in cross-validation.
        cv_mcc = cross_val_score(
            pipeline, x_train, y_train, cv=cv,
            scoring=make_scorer(matthews_corrcoef), n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / artifact_name)
        detailed = detailed_evaluation(pipeline, x_test, y_test)
        analyses[name] = detailed
        summary_rows.append({'ML Model Name': name, **detailed['metrics'],
                             'CV MCC mean': float(np.mean(cv_mcc)),
                             'CV MCC std': float(np.std(cv_mcc, ddof=1))})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / 'metrics.csv', index=False, float_format='%.6f')
    (RESULTS_DIR / 'detailed_results.json').write_text(
        json.dumps(analyses, indent=2), encoding='utf-8'
    )
    manifest = {
        'dataset': 'UCI Breast Cancer Wisconsin (Diagnostic)',
        'uci_url': 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
        'data_origin': 'scikit-learn packaged copy of the UCI WDBC dataset; original ID omitted',
        'dataset_sha256': sha256_file(DATASET_PATH),
        'test_sha256': sha256_file(TEST_DATA_PATH),
        'instances': int(len(original)), 'feature_count': len(feature_names),
        'feature_names': feature_names,
        'class_mapping': {'B': 0, 'M': 1},
        'positive_class': 'M (malignant), encoded as 1',
        'seed': RANDOM_SEED, 'test_fraction': TEST_FRACTION, 'cv_folds': CV_FOLDS,
        'training_count': len(x_train), 'test_count': len(x_test),
        'training_class_counts': {'B': int((y_train == 0).sum()), 'M': int((y_train == 1).sum())},
        'test_class_counts': {'B': int((y_test == 0).sum()), 'M': int((y_test == 1).sum())},
        'model_artifacts': {k: v[0] for k,v in MODEL_SPECIFICATIONS.items()},
        'algorithm_settings': {k: {'scaled': v[1], 'parameters': v[2].get_params()}
                               for k,v in MODEL_SPECIFICATIONS.items()},
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__,
    }
    (RESULTS_DIR / 'experiment.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8'
    )
    # Write the actual measured scores to the README after a successful lab run.
    from scripts.sync_readme_results import sync_readme_results
    sync_readme_results(summary, analyses, sklearn.__version__)

    print('Dataset:', manifest['instances'], 'samples x', manifest['feature_count'], 'predictors')
    print('Train/test:', len(x_train), '/', len(x_test), 'with seed', RANDOM_SEED)
    print('Target: 1=M (malignant); 0=B (benign)')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSaved five model pipelines and test_data.csv; no test samples were used to fit models.')


if __name__ == '__main__':
    main()
