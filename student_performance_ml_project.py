# Student Academic Performance Prediction - Mini Project
# Run in Google Colab or Jupyter Notebook.
# Dataset: UCI Student Performance (Mathematics)
# Target: pass=1 when G3 >= 10; fail=0 otherwise.
# Leakage control: G1, G2 and G3 are excluded from predictors.

import os, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
os.makedirs("results", exist_ok=True)

# Download official UCI mathematics dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
data_file = "student-mat.csv"
if not os.path.exists(data_file):
    urllib.request.urlretrieve(url, data_file)

df = pd.read_csv(data_file, sep=";")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nSummary:\n", df.describe(include="all").T)

# Exploratory analysis
df["result"] = np.where(df["G3"] >= 10, 1, 0)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="result")
plt.xticks([0,1], ["Fail", "Pass"])
plt.title("Pass/Fail Class Distribution")
plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=200)
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="absences", hue="result", bins=20, multiple="stack")
plt.title("Absences by Final Outcome")
plt.tight_layout()
plt.savefig("results/absences_by_outcome.png", dpi=200)
plt.show()

# Target is defined from G3. Exclude all grade columns to avoid leakage.
y = df["result"]
X = df.drop(columns=["G1", "G2", "G3", "result"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                                             max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE,
                                             class_weight="balanced", min_samples_leaf=2),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced",
                     random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = []
fitted = {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }
    results.append(row)
    fitted[name] = pipe
    print("\n", name)
    print(classification_report(y_test, pred, target_names=["Fail", "Pass"], zero_division=0))

scores = pd.DataFrame(results).sort_values("F1-score", ascending=False)
print("\nMODEL COMPARISON\n", scores.to_string(index=False))
scores.to_csv("results/model_comparison.csv", index=False)

# Confusion matrix and ROC curves for the model with highest F1
best_name = scores.iloc[0]["Model"]
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.show()

plt.figure(figsize=(7,5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=plt.gca())
plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig("results/roc_curves.png", dpi=200)
plt.show()

print("\nBest model by F1-score:", best_name)
print("Actual test-set size:", len(y_test))
print("Class distribution:\n", y.value_counts().rename(index={0:"Fail",1:"Pass"}))

# Optional single-record prediction: edit values to match the feature columns.
# Example: sample = X_test.iloc[[0]]
# print(best_model.predict(sample), best_model.predict_proba(sample))
