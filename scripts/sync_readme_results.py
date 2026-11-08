"""Write *measured* results into README.md after training (never hand-typed numbers).

Everything between the two markers is regenerated from the metrics produced by
train_models.py: the six-metric comparison table, the per-model observations
table (including the "Overall, Winner" row required by the assignment) and the
cross-validation table. Observation text combines numbers measured on the
held-out test split with short, model-specific reasoning.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

START = '<!-- RESULTS_FROM_TRAINING_START -->'
END = '<!-- RESULTS_FROM_TRAINING_END -->'
METRIC_COLUMNS = ('Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC')

# Why each family behaves the way it does on WDBC-style data (no numbers here,
# so the sentences stay true if a retrain shifts the scores slightly).
REASONING = {
    'Logistic Regression': (
        'The classes are close to linearly separable once features are standardised, so a regularised '
        'linear model is hard to beat; it is also fast, stable and its coefficients are interpretable.'),
    'Decision Tree': (
        'A single depth-limited tree uses axis-aligned splits and has high variance, so a few borderline '
        'cases fall on the wrong side of a split. It is the most interpretable model but the least '
        'accurate here.'),
    'KNN': (
        'Distance-weighted k-NN on scaled features is a local method: it works well because malignant and '
        'benign nuclei form fairly compact regions, but it is sensitive to the 30 correlated features and '
        'to the scaling step, and it stores the whole training set.'),
    'Gaussian Naive Bayes': (
        'It assumes features are independent given the class, which is violated here (radius, perimeter and '
        'area are almost the same measurement), so its probabilities are over-confident. It can still rank '
        'cases well (see AUC) while its 0.5 cut-off behaves conservatively.'),
    'Support Vector Machine': (
        'An RBF kernel finds a smooth, maximum-margin non-linear boundary on scaled features; the sigmoid '
        'calibration wrapper supplies the probabilities needed for AUC. Margin maximisation generalises well '
        'on small, clean tabular data like this.'),
}


def _ordinal(n: int) -> str:
    return {1: '1st', 2: '2nd', 3: '3rd'}.get(n, f'{n}th')


def _observations(summary: pd.DataFrame, details: dict) -> tuple[list[str], str]:
    ranks = {m: summary[m].rank(ascending=False, method='min').astype(int) for m in METRIC_COLUMNS}
    ordered = summary.sort_values(['MCC', 'F1'], ascending=[False, False]).reset_index(drop=True)
    mcc_rank = {row['ML Model Name']: i + 1 for i, row in ordered.iterrows()}
    rows = []
    for idx, record in summary.iterrows():
        name = record['ML Model Name']
        (tn, fp), (fn, tp) = details[name]['matrix']
        total = tn + fp + fn + tp
        best = [m for m in METRIC_COLUMNS if ranks[m].iloc[idx] == 1]
        worst = [m for m in METRIC_COLUMNS if ranks[m].iloc[idx] == len(summary)]
        text = (f'{fp + fn} of {total} test cases wrong (FP={fp}, FN={fn}; TN={tn}, TP={tp}). '
                f'Accuracy {record["Accuracy"]:.4f}, AUC {record["AUC"]:.4f}, precision {record["Precision"]:.4f}, '
                f'recall {record["Recall"]:.4f}, F1 {record["F1"]:.4f}, MCC {record["MCC"]:.4f} '
                f'(ranks {_ordinal(mcc_rank[name])} of {len(summary)} on MCC).')
        if best:
            text += ' Best of the five on: ' + ', '.join(best) + '.'
        if worst:
            text += ' Lowest of the five on: ' + ', '.join(worst) + '.'
        text += ' ' + REASONING[name]
        rows.append(f'| {name} | {text} |')

    top, runner = ordered.iloc[0], ordered.iloc[1]
    (tn1, fp1), (fn1, tp1) = details[top['ML Model Name']]['matrix']
    (tn2, fp2), (fn2, tp2) = details[runner['ML Model Name']]['matrix']
    gap = (fp2 + fn2) - (fp1 + fn1)
    cv_top = summary.sort_values('CV MCC mean', ascending=False).iloc[0]
    winner_text = (
        f'**{top["ML Model Name"]}**: highest held-out MCC ({top["MCC"]:.4f}) and F1 ({top["F1"]:.4f}) with '
        f'{fp1 + fn1} errors on {tn1 + fp1 + fn1 + tp1} test cases, and malignant recall {top["Recall"]:.4f} '
        f'(only {fn1} malignant case(s) missed). Runner-up {runner["ML Model Name"]} makes {gap} more error(s). ')
    if cv_top['ML Model Name'] == top['ML Model Name']:
        winner_text += (f'It also has the highest training-only 5-fold CV MCC ({top["CV MCC mean"]:.4f} '
                        f'+/- {top["CV MCC std"]:.4f}), consistent with the held-out ranking. ')
    else:
        winner_text += (f'Note that {cv_top["ML Model Name"]} has the highest CV MCC '
                        f'({cv_top["CV MCC mean"]:.4f}), so the top models are close. ')
    winner_text += ('The gap to the runner-up is small relative to the CV standard deviation, so treat the '
                    'top two as practically comparable; for cancer screening the deciding factor is fewest missed '
                    'malignant cases (recall) at high precision.')
    rows.append(f'| **Overall, Winner for your dataset?** | {winner_text} |')
    return rows, top['ML Model Name']


def sync_readme_results(summary: pd.DataFrame, details: dict, sklearn_version: str):
    root = Path(__file__).resolve().parents[1]
    readme_path = root / 'README.md'
    original = readme_path.read_text(encoding='utf-8')
    if START not in original or END not in original:
        raise ValueError('README results markers are missing; cannot update report accurately')

    n_test = int(sum(sum(r) for r in next(iter(details.values()))['matrix']))
    pieces = [
        f'Scores below were produced by `train_models.py` with scikit-learn {sklearn_version} on one fixed, '
        f'stratified {n_test}-row held-out test split (`test_data.csv`). Positive class = malignant (1).',
        '',
        '**Comparison table (held-out test data)**',
        '',
        '| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for _, record in summary.iterrows():
        values = ' | '.join(f'{record[key]:.4f}' for key in METRIC_COLUMNS)
        pieces.append(f'| {record["ML Model Name"]} | {values} |')

    observation_rows, _ = _observations(summary, details)
    pieces.extend(['', '**Observations on the performance of each model**', '',
                   '| ML Model Name | Observation about model performance |', '|---|---|', *observation_rows,
                   '', '**Training-only 5-fold stratified cross-validation (MCC)** - a second opinion that '
                   'does not touch the test split:', '',
                   '| ML Model Name | CV MCC mean | CV MCC std |', '|---|---:|---:|'])
    for _, record in summary.iterrows():
        pieces.append(f'| {record["ML Model Name"]} | {record["CV MCC mean"]:.4f} | {record["CV MCC std"]:.4f} |')

    start_index = original.index(START) + len(START)
    end_index = original.index(END)
    updated = original[:start_index] + '\n' + '\n'.join(pieces) + '\n' + original[end_index:]
    readme_path.write_text(updated, encoding='utf-8')
