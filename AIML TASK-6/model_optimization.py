import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("student.csv")

print("=" * 60)
print("First 5 Records")
print("=" * 60)
print(df.head())

# =====================================================
# Data Cleaning
# =====================================================

print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates()

print("\nDataset Shape:", df.shape)

# =====================================================
# Feature Engineering
# =====================================================

df["Pass"] = (
    (df["Attendance"] >= 75) &
    (df["Final Score"] >= 60)
).astype(int)

print("\nPass / Fail Count")
print(df["Pass"].value_counts())

# =====================================================
# Feature Selection
# =====================================================

X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score"
    ]
]

y = df["Pass"]

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data :", len(X_train))
print("Testing Data  :", len(X_test))

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# Previous Model
# =====================================================

print("\n" + "=" * 60)
print("Previous Logistic Regression")
print("=" * 60)

base_model = LogisticRegression(max_iter=1000)

base_model.fit(X_train_scaled, y_train)

base_prediction = base_model.predict(X_test_scaled)

base_probability = base_model.predict_proba(X_test_scaled)[:, 1]

base_accuracy = accuracy_score(y_test, base_prediction)
base_precision = precision_score(y_test, base_prediction)
base_recall = recall_score(y_test, base_prediction)
base_f1 = f1_score(y_test, base_prediction)
base_auc = roc_auc_score(y_test, base_probability)

print("Accuracy :", base_accuracy)
print("Precision:", base_precision)
print("Recall   :", base_recall)
print("F1 Score :", base_f1)
print("ROC AUC  :", base_auc)

# =====================================================
# Hyperparameter Tuning
# =====================================================

print("\nRunning GridSearchCV...")

parameters = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    parameters,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

print("\nBest Parameters")
print(grid.best_params_)

print("Best Cross Validation Accuracy")
print(grid.best_score_)

# =====================================================
# Tuned Model
# =====================================================

best_model = grid.best_estimator_

tuned_prediction = best_model.predict(X_test_scaled)

tuned_probability = best_model.predict_proba(X_test_scaled)[:, 1]

tuned_accuracy = accuracy_score(y_test, tuned_prediction)
tuned_precision = precision_score(y_test, tuned_prediction)
tuned_recall = recall_score(y_test, tuned_prediction)
tuned_f1 = f1_score(y_test, tuned_prediction)
tuned_auc = roc_auc_score(y_test, tuned_probability)

print("\n" + "=" * 60)
print("Tuned Logistic Regression")
print("=" * 60)

print("Accuracy :", tuned_accuracy)
print("Precision:", tuned_precision)
print("Recall   :", tuned_recall)
print("F1 Score :", tuned_f1)
print("ROC AUC  :", tuned_auc)

# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report")

print(classification_report(y_test, tuned_prediction))

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, tuned_prediction)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot(cmap="Blues")

plt.title("Tuned Logistic Regression Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.close()

# =====================================================
# Comparison Table
# =====================================================

comparison = pd.DataFrame({

    "Model": [
        "Previous Model",
        "Tuned Model"
    ],

    "Accuracy": [
        base_accuracy,
        tuned_accuracy
    ],

    "Precision": [
        base_precision,
        tuned_precision
    ],

    "Recall": [
        base_recall,
        tuned_recall
    ],

    "F1 Score": [
        base_f1,
        tuned_f1
    ],

    "ROC AUC": [
        base_auc,
        tuned_auc
    ]

})

print("\nComparison Table")
print(comparison)

comparison.to_csv("comparison_results.csv", index=False)

# =====================================================
# Comparison Chart
# =====================================================

plt.figure(figsize=(7,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Previous vs Tuned Model Accuracy")

plt.xlabel("Models")

plt.ylabel("Accuracy")

for i, value in enumerate(comparison["Accuracy"]):
    plt.text(i, value + 0.01, f"{value:.2f}", ha="center")

plt.tight_layout()

plt.savefig("comparison_chart.png")

plt.close()

print("\nFiles Generated Successfully!")
print("--------------------------------")
print("comparison_results.csv")
print("comparison_chart.png")
print("confusion_matrix.png")

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, tuned_probability)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {tuned_auc:.2f}")
plt.plot([0,1], [0,1], "r--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = tuned_prediction

results.to_csv("predictions.csv", index=False)

import numpy as np

importance = abs(best_model.coef_[0])

plt.figure(figsize=(6,4))

plt.bar(
    X.columns,
    importance
)

plt.title("Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")
plt.close()

print("\nFiles Generated Successfully!")
print("--------------------------------")
print("comparison_results.csv")
print("comparison_chart.png")
print("confusion_matrix.png")
print("roc_curve.png")
print("feature_importance.png")
print("predictions.csv")