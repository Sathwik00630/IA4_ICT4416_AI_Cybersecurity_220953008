# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import json, os, textwrap, pandas as pd

base = r"C:\Users\mathi\PycharmProjects\pythonProject4"
py_path = os.path.join(base, "nids_assignment_unsw_nb15.py")
ipynb_path = os.path.join(base, "nids_assignment_unsw_nb15.ipynb")
readme_path = os.path.join(base, "README_GitHub_Submission.md")
req_path = os.path.join(base, "requirements.txt")

py_code = r'''
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# =========================
# 1. File paths
# =========================
TRAIN_PATH = "UNSW_NB15_train_40k.csv"
TEST_PATH = "UNSW_NB15_test_10k.csv"

if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        "Place UNSW_NB15_train_40k.csv and UNSW_NB15_test_10k.csv "
        "in the same folder as this script / notebook."
    )


# =========================
# 2. Load data
# =========================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nTrain dtypes:\n", train_df.dtypes)
print("\nMissing values in train:\n", train_df.isnull().sum())
print("\nMissing values in test:\n", test_df.isnull().sum())

TARGET = "label"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

print("\nTrain class distribution:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(4))

print("\nTest class distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True).round(4))


# =========================
# 3. EDA
# =========================
os.makedirs("outputs", exist_ok=True)

# Class distribution
plt.figure(figsize=(6, 4))
train_df[TARGET].value_counts().sort_index().plot(kind="bar")
plt.title("Training Set Class Distribution")
plt.xlabel("Label (0=Normal, 1=Attack)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.show()

# Categorical feature distributions
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    top_counts = train_df[col].value_counts().head(15)
    top_counts.plot(kind="bar")
    plt.title(f"Top Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_distribution.png")
    plt.show()

# Numerical feature histograms
for col in numerical_cols:
    plt.figure(figsize=(7, 4))
    plt.hist(train_df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_hist.png")
    plt.show()

# Boxplots for outlier inspection
for col in numerical_cols:
    plt.figure(figsize=(7, 2.8))
    plt.boxplot(train_df[col], vert=False)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_boxplot.png")
    plt.show()

# Simple grouped summaries
grouped_means = train_df.groupby(TARGET)[numerical_cols].mean().T
grouped_means.columns = ["Normal (0)", "Attack (1)"]
print("\nMean of numerical features by class:\n")
print(grouped_means.round(3))

grouped_means.to_csv("outputs/grouped_feature_means.csv")


# =========================
# 4. Preprocessing helpers
# =========================
class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using the IQR rule learned from the training data.
    """
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns.tolist()
        self.bounds_ = {}

        for col in self.columns_:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for col in self.columns_:
            lower, upper = self.bounds_[col]
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X


def build_preprocessor(scale_numeric=True, dense_output=False, include_categorical=True):
    num_pipeline_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
    ]
    if scale_numeric:
        num_pipeline_steps.append(("scaler", StandardScaler()))

    transformers = [
        ("num", Pipeline(num_pipeline_steps), numerical_cols)
    ]

    if include_categorical:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.0 if dense_output else 0.3
    )


# =========================
# 5. Models
# =========================
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "preprocessor_args": dict(scale_numeric=True, dense_output=True, include_categorical=True),
    },
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=9),
        "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
    },
    "SVM": {
        "model": LinearSVC(class_weight="balanced", max_iter=4000),
        "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=True),
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=True),
    },
    "Gradient Boosting": {
        # Fast alternative to classic GradientBoostingClassifier for this assignment
        "model": HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=120,
            learning_rate=0.1,
            random_state=42
        ),
        # HistGradientBoosting does not accept sparse OHE directly; use numeric features only
        "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=False),
    },
    "Deep Neural Network": {
        "model": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=30,
            early_stopping=True,
            random_state=42
        ),
        "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
    },
}


# =========================
# 6. Train + evaluate
# =========================
results = []
trained_pipelines = {}

for model_name, config in models.items():
    print(f"\nTraining: {model_name}")
    preprocessor = build_preprocessor(**config["preprocessor_args"])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", config["model"]),
    ])

    pipeline.fit(X_train, y_train)
    trained_pipelines[model_name] = pipeline

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)
    else:
        y_score = y_pred

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F2 Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "F2 Macro": fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0),
        "PR AUC": average_precision_score(y_test, y_score),
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"outputs/confusion_matrix_{safe_name}.png")
    plt.show()

    # PR curve
    try:
        plt.figure(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_score)
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(f"outputs/pr_curve_{safe_name}.png")
        plt.show()
    except Exception as e:
        print(f"PR curve could not be plotted for {model_name}: {e}")

results_df = pd.DataFrame(results).sort_values(
    by=["F2 Score", "Recall", "PR AUC"],
    ascending=False
).reset_index(drop=True)

print("\nFinal model comparison:\n")
print(results_df.round(4))

results_df.to_csv("outputs/model_comparison_results.csv", index=False)


# =========================
# 7. Automatic conclusions
# =========================
best_model = results_df.iloc[0]["Model"]
best_f2 = results_df.iloc[0]["F2 Score"]
best_recall = results_df.iloc[0]["Recall"]

print("\n===== INTERPRETATION TEMPLATE =====")
print(f"Best model by F2 Score: {best_model}")
print(f"Best F2 Score         : {best_f2:.4f}")
print(f"Best Recall           : {best_recall:.4f}")

print("""
Suggested report discussion points:
1. The training data is moderately imbalanced, while the testing data has a noticeably different
   class proportion. This indicates dataset shift, so recall, F2, and PR-AUC are more meaningful
   than accuracy alone.
2. One-hot encoding was applied to proto, state, and service because they are categorical.
3. Numerical features were median-imputed, outliers were capped using IQR, and scaling was
   applied to distance-based / margin-based models such as KNN, SVM, Logistic Regression,
   and the neural network.
4. A higher recall is especially important in intrusion detection because missing an attack is often
   more costly than incorrectly flagging a normal packet flow.
5. F2 Score is appropriate here because it weights recall more heavily than precision.
6. In a real-world deployment, the final model should be monitored continuously because traffic
   patterns may change over time, causing performance drift.
""")

print("\nAll plots and result files have been saved inside the outputs/ folder.")
'''.strip()


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True)
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True)
    }

notebook_cells = [
    md_cell("""
    # Network Intrusion Detection System using UNSW-NB15
    This notebook is prepared for **ICT4416 AI in Cybersecurity – IA4**.

    It covers:
    - Exploratory Data Analysis (EDA)
    - Data preprocessing
    - Multiple ML models
    - A basic Deep Neural Network
    - Evaluation using Accuracy, Precision, Recall, F2, F2-Macro, PR-AUC
    - Confusion matrices and conclusions
    """),
    md_cell("""
    ## Notes
    - Put `UNSW_NB15_train_40k.csv` and `UNSW_NB15_test_10k.csv` in the same folder as this notebook.
    - The assignment PDF asks for binary classification of network traffic as **Normal (0)** or **Attack (1)**.
    - F2 score is included because it gives more importance to recall, which is critical in intrusion detection.
    """),
    code_cell("""
    import warnings
    warnings.filterwarnings("ignore")

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        fbeta_score, average_precision_score,
        confusion_matrix, ConfusionMatrixDisplay,
        PrecisionRecallDisplay
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    """),
    code_cell("""
    TRAIN_PATH = "UNSW_NB15_train_40k.csv"
    TEST_PATH = "UNSW_NB15_test_10k.csv"

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    TARGET = "label"
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\\nCategorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)
    print("\\nTrain class distribution:")
    print(y_train.value_counts(normalize=True).round(4))
    print("\\nTest class distribution:")
    print(y_test.value_counts(normalize=True).round(4))
    print("\\nMissing values in train:")
    print(train_df.isnull().sum())
    """),
    md_cell("""
    ## EDA
    """),
    code_cell("""
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    train_df[TARGET].value_counts().sort_index().plot(kind="bar")
    plt.title("Training Set Class Distribution")
    plt.xlabel("Label (0=Normal, 1=Attack)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png")
    plt.show()
    """),
    code_cell("""
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        train_df[col].value_counts().head(15).plot(kind="bar")
        plt.title(f"Top Categories in {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"outputs/{col}_distribution.png")
        plt.show()
    """),
    code_cell("""
    for col in numerical_cols:
        plt.figure(figsize=(7, 4))
        plt.hist(train_df[col], bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"outputs/{col}_hist.png")
        plt.show()
    """),
    code_cell("""
    grouped_means = train_df.groupby(TARGET)[numerical_cols].mean().T
    grouped_means.columns = ["Normal (0)", "Attack (1)"]
    grouped_means.round(3)
    """),
    md_cell("""
    ## Preprocessing
    - Median imputation for numerical columns
    - Most-frequent imputation for categorical columns
    - IQR-based outlier capping
    - One-hot encoding for `proto`, `state`, `service`
    - Standard scaling for models that depend on feature magnitude
    """),
    code_cell("""
    class IQRClipper(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            X = pd.DataFrame(X).copy()
            self.columns_ = X.columns.tolist()
            self.bounds_ = {}
            for col in self.columns_:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            return self

        def transform(self, X):
            X = pd.DataFrame(X, columns=self.columns_).copy()
            for col in self.columns_:
                lower, upper = self.bounds_[col]
                X[col] = X[col].clip(lower=lower, upper=upper)
            return X

    def build_preprocessor(scale_numeric=True, dense_output=False, include_categorical=True):
        num_pipeline_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", IQRClipper()),
        ]
        if scale_numeric:
            num_pipeline_steps.append(("scaler", StandardScaler()))

        transformers = [
            ("num", Pipeline(num_pipeline_steps), numerical_cols)
        ]

        if include_categorical:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

        return ColumnTransformer(
            transformers=transformers,
            sparse_threshold=0.0 if dense_output else 0.3
        )
    """),
    md_cell("""
    ## Model Building
    """),
    code_cell("""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1500, class_weight="balanced"),
            "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "preprocessor_args": dict(scale_numeric=True, dense_output=True, include_categorical=True),
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=9),
            "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
        },
        "SVM": {
            "model": LinearSVC(class_weight="balanced", max_iter=4000),
            "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
            "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=True),
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=14,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=True),
        },
        "Gradient Boosting": {
            "model": HistGradientBoostingClassifier(
                max_depth=8,
                max_iter=120,
                learning_rate=0.1,
                random_state=42
            ),
            "preprocessor_args": dict(scale_numeric=False, dense_output=False, include_categorical=False),
        },
        "Deep Neural Network": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=30,
                early_stopping=True,
                random_state=42
            ),
            "preprocessor_args": dict(scale_numeric=True, dense_output=False, include_categorical=True),
        },
    }
    """),
    code_cell("""
    results = []
    trained_pipelines = {}

    for model_name, config in models.items():
        print(f"Training: {model_name}")
        preprocessor = build_preprocessor(**config["preprocessor_args"])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["model"]),
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)
        else:
            y_score = y_pred

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F2 Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            "F2 Macro": fbeta_score(y_test, y_pred, beta=2, average="macro", zero_division=0),
            "PR AUC": average_precision_score(y_test, y_score),
        })

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.show()

        try:
            plt.figure(figsize=(5, 4))
            PrecisionRecallDisplay.from_predictions(y_test, y_score)
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot PR curve for {model_name}: {e}")
    """),
    code_cell("""
    results_df = pd.DataFrame(results).sort_values(
        by=["F2 Score", "Recall", "PR AUC"],
        ascending=False
    ).reset_index(drop=True)

    results_df
    """),
    md_cell("""
    ## Final Discussion Points
    Use these points in your report:
    1. The test set has a different class balance from the training set, so this is a realistic intrusion-detection setting with distribution shift.
    2. Recall and F2 are especially important because missing attacks is usually more dangerous than raising extra alarms.
    3. Tree-based and ensemble models often handle nonlinear traffic behavior better, while linear models are easier to interpret.
    4. Scaling improved distance-based models such as KNN and margin-based models such as SVM.
    5. Continuous monitoring is required in real deployments because traffic behavior changes over time.
    """)
]

notebook = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

readme = """
# ICT4416 IA4 - AI in Cybersecurity
This repository contains a complete solution template for the UNSW-NB15 Network Intrusion Detection System assignment.

## Files
- `nids_assignment_unsw_nb15.ipynb` - Jupyter Notebook version
- `nids_assignment_unsw_nb15.py` - PyCharm / Python script version
- `requirements.txt` - required Python libraries

## Dataset files required
Place these files in the same folder:
- `UNSW_NB15_train_40k.csv`
- `UNSW_NB15_test_10k.csv`

## How to run in Jupyter
1. Open the folder in Jupyter Notebook or VS Code / PyCharm.
2. Open `nids_assignment_unsw_nb15.ipynb`.
3. Run all cells from top to bottom.

## How to run in PyCharm
1. Open the project folder in PyCharm.
2. Install the required libraries from `requirements.txt`.
3. Run `nids_assignment_unsw_nb15.py`.

## Output
The code creates an `outputs/` folder containing:
- EDA plots
- confusion matrices
- precision-recall curves
- model comparison CSV
- grouped feature summary CSV

## GitHub upload
Upload the notebook/script, requirements file, and output screenshots/results to your GitHub repository, then paste that GitHub link into your Word/PDF submission document.
""".strip()

requirements = """
pandas
numpy
matplotlib
scikit-learn
jupyter
notebook
""".strip()

with open(py_path, "w", encoding="utf-8") as f:
    f.write(py_code)

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

with open(req_path, "w", encoding="utf-8") as f:
    f.write(requirements)

print("Created files:")
for p in [py_path, ipynb_path, readme_path, req_path]:
    print("-", p)
