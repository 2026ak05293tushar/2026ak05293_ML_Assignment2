"""Interactive, evaluation-only Streamlit app: never trains on uploaded CSV."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay

from ml_utils import (
    PROJECT_ROOT, MODEL_DIR, RESULTS_DIR, TEST_DATA_PATH, TARGET_COLUMN, CLASS_NAMES,
    detailed_evaluation, evaluate_classifier, load_labeled_csv,
)

st.set_page_config(page_title='WDBC | Model Evaluation Studio', page_icon='📊', layout='wide')

@st.cache_resource(show_spinner='Loading trained classifiers…')
def load_trained_models():
    """Load serialized *fitted* pipelines without importing training procedures."""
    with (RESULTS_DIR / 'experiment.json').open('r', encoding='utf-8') as stream:
        metadata = json.load(stream)
    # Fail BEFORE unpickling models trained with another sklearn version.
    trained_with = metadata.get('scikit_learn')
    running_with = sklearn.__version__
    if trained_with != running_with:
        raise RuntimeError(
            f'Model version mismatch: saved with scikit-learn {trained_with}, '
            f'currently running {running_with}. Run `python train_models.py` '
            'in this environment, then restart Streamlit. Do not ignore '
            'InconsistentVersionWarning.'
        )
    models = {name: joblib.load(MODEL_DIR / filename)
              for name, filename in metadata['model_artifacts'].items()}
    return metadata, models

try:
    metadata, trained_models = load_trained_models()
except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
    st.error(f'Model artifacts unavailable: {error}. Execute `python train_models.py` first.')
    st.stop()

st.title('Breast Cancer Wisconsin · Classifier Evaluation')
st.caption('BITS WILP AIML | Assignment 2 learning implementation · UCI WDBC · non-clinical demonstration')
st.info('This is a statistical classification demonstration, **not a medical diagnostic tool**.')

with st.sidebar:
    st.header('Experiment contract')
    st.write(f"**Observations:** {metadata['instances']}")
    st.write(f"**Numeric features:** {metadata['feature_count']}")
    st.write(f"**Training rows:** {metadata['training_count']}")
    st.write(f"**Held-out test rows:** {metadata['test_count']}")
    st.write(f"**Random seed:** {metadata['seed']}")
    st.write('**Positive class:** M = malignant')
    st.markdown('[Original dataset at UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)')

st.subheader('1 · Upload a labeled test CSV')
st.write('Upload a CSV with all 30 numeric predictor columns and a `diagnosis` column containing **B** or **M**. '
         'Training data are never accepted or used for fitting in this application.')
col_upload, col_help = st.columns([2, 1])
with col_upload:
    csv_file = st.file_uploader('Choose CSV containing test observations and true diagnoses', type=['csv'])
with col_help:
    st.download_button('Download test CSV example', data=(PROJECT_ROOT / 'test_data.csv').read_bytes(),
                       file_name='test_data.csv', mime='text/csv', width='stretch')

use_sample = st.checkbox('No file handy? Evaluate on the bundled test_data.csv (the 114 held-out rows)')
if csv_file is None and not use_sample:
    st.warning('Upload a test CSV (or tick the box above) to display the six metrics, model comparison, and confusion matrix.')
    with st.expander('Data dictionary / required feature names'):
        st.write(pd.DataFrame({'Required column': [*metadata['feature_names'], TARGET_COLUMN]}))
    st.stop()

try:
    features, truth = load_labeled_csv(csv_file if csv_file is not None else TEST_DATA_PATH, metadata['feature_names'])
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error('Unable to evaluate CSV: ' + str(error))
    st.stop()

if features.shape[0] > 10000:
    st.error('Limit testing to 10,000 rows for this educational Community Cloud app.')
    st.stop()

st.success(f'Validated {len(features)} labeled observations with {features.shape[1]} numeric features.')
if truth.nunique() < 2:
    st.warning('AUC cannot be calculated for this upload because its true labels contain only one class.')

st.subheader('2 · Choose one of the five saved models')
selected_name = st.selectbox('Trained classifier', list(trained_models))
selected_model = trained_models[selected_name]

st.subheader('3 · Required evaluation metrics')
results = detailed_evaluation(selected_model, features, truth)
metrics = results['metrics']
columns = st.columns(6)
for col, metric_name in zip(columns, ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']):
    with col:
        value = metrics[metric_name]
        st.metric(metric_name, f'{value:.4f}' if value is not None else 'N/A')

plot_col, report_col = st.columns([1, 1], gap='large')
with plot_col:
    st.markdown('**Confusion matrix** (rows: actual, columns: predicted)')
    matrix = np.asarray(results['matrix'], dtype=int)
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    ConfusionMatrixDisplay(matrix, display_labels=['Benign', 'Malignant']).plot(
        ax=axis, colorbar=False, cmap='Blues', values_format='d'
    )
    axis.set_title(selected_name)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.caption('Top-left: TN · top-right: FP · bottom-left: FN · bottom-right: TP')
with report_col:
    st.markdown('**Per-class classification report**')
    report_table = pd.DataFrame(results['report']).T
    st.dataframe(report_table.round(4), width='stretch')
    st.markdown('**Class balance in uploaded data**')
    counts = truth.value_counts().reindex([0, 1], fill_value=0)
    st.bar_chart(pd.DataFrame({'Count': counts.values}, index=['Benign', 'Malignant']))

st.subheader('4 · Compare all five models on this SAME uploaded test CSV')
comparison_rows = []
for name, fitted_model in trained_models.items():
    comparison_rows.append({'ML Model Name': name, **evaluate_classifier(fitted_model, features, truth)})
comparison = pd.DataFrame(comparison_rows)
st.dataframe(comparison.style.format({c: '{:.4f}' for c in ['Accuracy','AUC','Precision','Recall','F1','MCC']}, na_rep='N/A'),
             width='stretch', hide_index=True)

predicted = selected_model.predict(features)
export_frame = features.copy()
export_frame[TARGET_COLUMN] = truth.map({0: 'B', 1: 'M'}).to_numpy()
export_frame['predicted_diagnosis'] = pd.Series(predicted).map({0: 'B', 1: 'M'}).to_numpy()
export_frame['probability_malignant'] = selected_model.predict_proba(features)[:, list(selected_model.classes_).index(1)]
st.download_button('Download selected-model predictions (CSV)', export_frame.to_csv(index=False),
                   file_name='wdbc_predictions.csv', mime='text/csv')

with st.expander('What each score means'):
    st.markdown('''
- **Accuracy**: correct predictions / total predictions.
- **Precision (malignant)**: share of predicted malignant observations that actually are malignant.
- **Recall (malignant)**: share of actual malignant observations detected by the classifier.
- **F1**: harmonic mean of malignant-class precision and recall.
- **MCC**: balanced correlation from -1 (opposite) to +1 (perfect), useful with uneven classes.
- **ROC AUC**: ranking quality based on malignant-class probability scores. Undefined when the actual uploaded labels include only one class.

All models were fitted separately before deployment. **Uploading a CSV never refits a model.**
''')
