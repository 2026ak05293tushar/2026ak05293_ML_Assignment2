# Dataset origin and transformation

**Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), Dataset ID 17.
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Packaged copy:** `sklearn.datasets.load_breast_cancer(as_frame=True)` provides a bundled UCI-origin numerical version with 569 observations and 30 features. It has the same data records and the documented class mapping `0=malignant, 1=benign`.

**For this learning project:** Replaced spaces in the 30 feature names with underscores and appended `diagnosis` encoded as the UCI letter label `M` for malignant or `B` for benign. The original ID column is not included. The model training code maps B->0, M->1, deliberately making the malignant class positive for all metrics.

**Why not download directly from UCI at training time?** To keep the project deterministic and runnable in restricted BITS lab environments without external network access; the official UCI repository remains the dataset's source of record. A local bundled copy reduces deployment failure.

`results/experiment.json` records the SHA-256 checksum of the exact local CSV used during training. The original raw UCI file is `wdbc.data` and is not falsely claimed to have been downloaded here.
