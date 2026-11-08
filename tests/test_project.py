"""Smoke/regression tests run against shipped trained artifacts and CSVs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_utils import PROJECT_ROOT, TEST_DATA_PATH, RESULTS_DIR, MODEL_DIR, load_labeled_csv, evaluate_classifier


def setup_files():
    meta = json.loads((RESULTS_DIR / 'experiment.json').read_text(encoding='utf-8'))
    features, targets = load_labeled_csv(TEST_DATA_PATH, meta['feature_names'])
    return meta, features, targets


def test_dataset_requirements():
    meta, features, truth = setup_files()
    data = pd.read_csv(PROJECT_ROOT / 'data' / 'uci_wdbc_dataset.csv')
    assert data.shape == (569, 31)
    assert len(features.columns) == 30
    assert len(features) == 114
    assert set(truth.unique()) == {0, 1}
    assert set(data['diagnosis']) == {'B', 'M'}


def test_models_are_fitted_and_scores_reproduce():
    meta, features, truth = setup_files()
    from_results = pd.read_csv(RESULTS_DIR / 'metrics.csv').set_index('ML Model Name')
    assert len(meta['model_artifacts']) == 5
    for name, filename in meta['model_artifacts'].items():
        classifier = joblib.load(MODEL_DIR / filename)
        calculated = evaluate_classifier(classifier, features, truth)
        for metric_name in ('Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'):
            assert np.isclose(calculated[metric_name], from_results.loc[name, metric_name], atol=1e-6)


def test_validation_rejects_missing_target(tmp_path):
    meta, features, truth = setup_files()
    bad_path = tmp_path / 'bad.csv'
    features.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match='Required columns'):
        load_labeled_csv(bad_path, meta['feature_names'])


def test_validation_rejects_unexpected_target(tmp_path):
    meta, features, truth = setup_files()
    wrong = features.copy()
    wrong['diagnosis'] = 'unknown'
    file_path = tmp_path / 'bad_targets.csv'
    wrong.to_csv(file_path, index=False)
    with pytest.raises(ValueError, match='only'):
        load_labeled_csv(file_path, meta['feature_names'])


def test_holdout_partition_does_not_overlap_training():
    meta, features, truth = setup_files()
    training = pd.read_csv(RESULTS_DIR / 'training_partition.csv')
    test = pd.read_csv(TEST_DATA_PATH)
    key_features = meta['feature_names']
    # Numeric representation in exported CSV is rounded equally for both splits.
    train_keys = set(map(tuple, training[key_features].round(7).to_numpy()))
    test_keys = set(map(tuple, test[key_features].round(7).to_numpy()))
    assert train_keys.isdisjoint(test_keys)


def test_version_and_svm_calibration():
    """Artifacts, once generated, must match runtime version and not use deprecated SVC probabilities."""
    import sklearn
    from sklearn.calibration import CalibratedClassifierCV
    from train_models import MODEL_SPECIFICATIONS
    meta, _, _ = setup_files()
    assert meta['scikit_learn'] == sklearn.__version__
    svm_spec = MODEL_SPECIFICATIONS['Support Vector Machine'][2]
    assert isinstance(svm_spec, CalibratedClassifierCV)
    assert svm_spec.ensemble is False
    # scikit-learn >= 1.9 reports the deprecated parameter as the string 'deprecated';
    # older versions report False. Either way it must NOT be enabled (True).
    assert svm_spec.estimator.get_params()['probability'] in (False, 'deprecated')


def test_readme_follows_assignment_structure():
    """README must carry the sections and tables the assignment grades."""
    text = (PROJECT_ROOT / 'README.md').read_text(encoding='utf-8')
    for heading in ('## a. Problem statement', '## b. Dataset description',
                    '## c. GitHub repository link', '## d. Models used'):
        assert heading in text
    assert '| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |' in text
    assert '| ML Model Name | Observation about model performance |' in text
    assert 'Overall, Winner for your dataset?' in text
    meta, _, _ = setup_files()
    for name in meta['model_artifacts']:
        assert text.count(f'| {name} |') >= 3  # hyperparameters, metrics and observations rows


def test_all_five_model_files_ship_with_repo():
    meta, _, _ = setup_files()
    for filename in meta['model_artifacts'].values():
        assert (MODEL_DIR / filename).is_file()
