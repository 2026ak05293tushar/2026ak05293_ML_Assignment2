# Breast Cancer Wisconsin (Diagnostic) - Five-Model Classification and Streamlit App

M.Tech (AIML), BITS Pilani WILP | Machine Learning | Assignment 2

## a. Problem statement

Predict whether a breast mass is **malignant or benign** from 30 numeric measurements of cell nuclei, and compare five standard classifiers on the same data: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine. Each model is trained on an identical training partition and evaluated on the same held-out test partition with six metrics: Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). An interactive Streamlit app loads the saved models, accepts a labelled test CSV and shows the metrics and confusion matrix without retraining. This is a statistical classification exercise, not a medical diagnostic tool.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, [Breast Cancer Wisconsin (Diagnostic), dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The local file `data/uci_wdbc_dataset.csv` comes from scikit-learn's bundled copy of this UCI dataset (see `data/DATA_ORIGIN.md`); the patient ID column is dropped and column names use underscores.
- **Size:** 569 instances and 30 numeric input features (assignment minimum: 500 instances, 12 features). No missing values.
- **Task:** binary classification. Target `diagnosis`: B = benign (357, 62.7%), M = malignant (212, 37.3%).
- **Features:** 10 measurements of each cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as mean, standard error and "worst" value.
- **Encoding:** 0 = benign, **1 = malignant**; malignant is the positive class for precision, recall, F1 and AUC.
- **Split:** stratified 80/20, `random_state=42`: 455 training rows and 114 test rows (72 benign, 42 malignant). The test rows are saved as `test_data.csv`.
- **Preprocessing:** a median imputer (no values are missing) and, for Logistic Regression, KNN and SVM, a `StandardScaler`. Both are fitted inside a scikit-learn `Pipeline` on training data only, so nothing leaks from the test split.

## c. GitHub repository link

https://github.com/2026ak05293tushar/2026ak05293_ML_Assignment2

Live Streamlit app: http://localhost:8501/   \\No access to streamlit cloud was granted
\\put local run app in Github repository path above named  --> **WDBC _ Model Evaluation Studio.html**

## d. Models used

| Classifier | Hyperparameters | Scaling |
|---|---|---|
| Logistic Regression | `max_iter=5000`, `random_state=42` | Yes |
| Decision Tree | `max_depth=5`, `min_samples_leaf=3`, `random_state=42` | No |
| KNN | `n_neighbors=7`, `weights='distance'` | Yes |
| Gaussian Naive Bayes | defaults | No |
| Support Vector Machine | RBF kernel, `C=1.0`, sigmoid calibration (`CalibratedClassifierCV`, 5-fold, `ensemble=False`) to obtain probabilities for AUC | Yes |

<!-- RESULTS_FROM_TRAINING_START -->
Scores below were produced by `train_models.py` with scikit-learn 1.9.1 on one fixed, stratified 114-row held-out test split (`test_data.csv`). Positive class = malignant (1).

**Comparison table (held-out test data)**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9835 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |

**Observations on the performance of each model**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 4 of 114 test cases wrong (FP=1, FN=3; TN=71, TP=39). Accuracy 0.9649, AUC 0.9960, precision 0.9750, recall 0.9286, F1 0.9512, MCC 0.9245 (ranks 2nd of 5 on MCC). Best of the five on: AUC. The classes are close to linearly separable once features are standardised, so a regularised linear model is hard to beat; it is also fast, stable and its coefficients are interpretable. |
| Decision Tree | 11 of 114 test cases wrong (FP=2, FN=9; TN=70, TP=33). Accuracy 0.9035, AUC 0.9471, precision 0.9429, recall 0.7857, F1 0.8571, MCC 0.7926 (ranks 5th of 5 on MCC). Lowest of the five on: Accuracy, AUC, Precision, Recall, F1, MCC. A single depth-limited tree uses axis-aligned splits and has high variance, so a few borderline cases fall on the wrong side of a split. It is the most interpretable model but the least accurate here. |
| KNN | 5 of 114 test cases wrong (FP=1, FN=4; TN=71, TP=38). Accuracy 0.9561, AUC 0.9835, precision 0.9744, recall 0.9048, F1 0.9383, MCC 0.9058 (ranks 3rd of 5 on MCC). Distance-weighted k-NN on scaled features is a local method: it works well because malignant and benign nuclei form fairly compact regions, but it is sensitive to the 30 correlated features and to the scaling step, and it stores the whole training set. |
| Gaussian Naive Bayes | 7 of 114 test cases wrong (FP=0, FN=7; TN=72, TP=35). Accuracy 0.9386, AUC 0.9934, precision 1.0000, recall 0.8333, F1 0.9091, MCC 0.8715 (ranks 4th of 5 on MCC). Best of the five on: Precision. It assumes features are independent given the class, which is violated here (radius, perimeter and area are almost the same measurement), so its probabilities are over-confident. It can still rank cases well (see AUC) while its 0.5 cut-off behaves conservatively. |
| Support Vector Machine | 3 of 114 test cases wrong (FP=1, FN=2; TN=71, TP=40). Accuracy 0.9737, AUC 0.9947, precision 0.9756, recall 0.9524, F1 0.9639, MCC 0.9433 (ranks 1st of 5 on MCC). Best of the five on: Accuracy, Recall, F1, MCC. An RBF kernel finds a smooth, maximum-margin non-linear boundary on scaled features; the sigmoid calibration wrapper supplies the probabilities needed for AUC. Margin maximisation generalises well on small, clean tabular data like this. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**: highest held-out MCC (0.9433) and F1 (0.9639) with 3 errors on 114 test cases, and malignant recall 0.9524 (only 2 malignant case(s) missed). Runner-up Logistic Regression makes 1 more error(s). It also has the highest training-only 5-fold CV MCC (0.9492 +/- 0.0194), consistent with the held-out ranking. The gap to the runner-up is small relative to the CV standard deviation, so treat the top two as practically comparable; for cancer screening the deciding factor is fewest missed malignant cases (recall) at high precision. |

**Training-only 5-fold stratified cross-validation (MCC)** - a second opinion that does not touch the test split:

| ML Model Name | CV MCC mean | CV MCC std |
|---|---:|---:|
| Logistic Regression | 0.9444 | 0.0350 |
| Decision Tree | 0.8408 | 0.0533 |
| KNN | 0.9394 | 0.0267 |
| Gaussian Naive Bayes | 0.8737 | 0.0638 |
| Support Vector Machine | 0.9492 | 0.0194 |
<!-- RESULTS_FROM_TRAINING_END -->

How to read the metrics: Accuracy = (TP+TN)/all; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = harmonic mean of precision and recall; MCC = correlation between predicted and true labels, from -1 to +1, robust to class imbalance; AUC = area under the ROC curve computed from the malignant-class probability. A false negative (a missed malignant case) is the costly error here, so recall matters most.

## Streamlit app

`app.py` provides: CSV upload of test data (or a checkbox to use the bundled `test_data.csv`), a model-selection dropdown, the six evaluation metrics, a confusion matrix and per-class classification report, a side-by-side table of all five models on the same uploaded data, and a download of the predictions. The app only **loads** the saved pipelines in `model/` with `joblib`; it never fits anything. The CSV needs the 30 feature columns (names as in `test_data.csv`) plus `diagnosis` with values B or M.

## Run it

```bash
python -m pip install -r requirements.txt
python train_models.py        # optional: regenerates model/, results/, test_data.csv and the tables above
python -m pip install -r requirements-dev.txt
python -m pytest tests -q     # optional checks
streamlit run app.py
```

The saved models were trained with **scikit-learn 1.9.1** (pinned in `requirements.txt`; needs Python 3.11 or newer). A model file must be loaded with the same scikit-learn version that saved it, so the app checks `results/experiment.json` and asks you to rerun `train_models.py` if the versions differ.

## Repository structure

```text
app.py                      Streamlit app (evaluation only)
train_models.py             trains the five pipelines, writes model/, results/, test_data.csv
ml_utils.py                 CSV validation and metric functions
requirements.txt            runtime dependencies for Streamlit Community Cloud
test_data.csv               114 labelled held-out test rows
data/                       dataset (uci_wdbc_dataset.csv) and DATA_ORIGIN.md
model/                      five fitted pipelines (.joblib)
results/                    metrics.csv, detailed_results.json, experiment.json, training_partition.csv
notebooks/                  WDBC_Five_Classifiers_Learning.ipynb
tests/test_project.py       automated checks
scripts/                    README sync and submission-PDF builder
```

## Limitations

- One fixed 114-row test split is a limited estimate; a difference of one or two test cases between models is not strong evidence.
- Hyperparameters are sensible fixed values, not exhaustively tuned.
- Results are for a classroom comparison and say nothing about clinical performance.

## References

1. UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
2. scikit-learn documentation: [load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [model persistence](https://scikit-learn.org/stable/model_persistence.html).
3. Streamlit: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
