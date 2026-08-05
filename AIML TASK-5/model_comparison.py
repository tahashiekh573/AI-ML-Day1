import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
# Create Target Column
# =====================================================

df["Pass"] = (
    (df["Attendance"] >= 75) &
    (df["Final Score"] >= 60)
).astype(int)

print("\nPass / Fail Count")
print(df["Pass"].value_counts())

# =====================================================
# Features & Target
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
# Logistic Regression
# =====================================================

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

# =====================================================
# Decision Tree
# =====================================================

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

# =====================================================
# Logistic Regression Results
# =====================================================

print("\n========== Logistic Regression ==========")

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

# =====================================================
# Decision Tree Results
# =====================================================

print("\n========== Decision Tree ==========")

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)

print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1 Score :", dt_f1)

# =====================================================
# Confusion Matrices
# =====================================================

cm_lr = confusion_matrix(y_test, lr_pred, labels=[0,1])
cm_dt = confusion_matrix(y_test, dt_pred, labels=[0,1])

print("\nLogistic Regression Confusion Matrix")
print(cm_lr)

print("\nDecision Tree Confusion Matrix")
print(cm_dt)

# =====================================================
# Classification Reports
# =====================================================

print("\nClassification Report - Logistic Regression")
print(classification_report(y_test, lr_pred))

print("\nClassification Report - Decision Tree")
print(classification_report(y_test, dt_pred))

# =====================================================
# Comparison Table
# =====================================================

comparison = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree"
    ],

    "Accuracy":[
        lr_accuracy,
        dt_accuracy
    ],

    "Precision":[
        lr_precision,
        dt_precision
    ],

    "Recall":[
        lr_recall,
        dt_recall
    ],

    "F1 Score":[
        lr_f1,
        dt_f1
    ]

})

print("\n==============================")
print("Model Comparison")
print("==============================")
print(comparison)

comparison.to_csv(
    "comparison_results.csv",
    index=False
)

# =====================================================
# Accuracy Chart
# =====================================================

plt.figure(figsize=(7,5))

bars = plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylim(0,1.1)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        height+0.02,
        f"{height:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig("comparison_chart.png")

plt.close()

# =====================================================
# Logistic Confusion Matrix
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Fail","Pass"]
)

disp.plot()

plt.title("Logistic Regression Confusion Matrix")

plt.savefig("logistic_confusion_matrix.png")

plt.close()

# =====================================================
# Decision Tree Confusion Matrix
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Fail","Pass"]
)

disp.plot()

plt.title("Decision Tree Confusion Matrix")

plt.savefig("decision_tree_confusion_matrix.png")

plt.close()

print("\nFiles Generated Successfully!")
print("- comparison_results.csv")
print("- comparison_chart.png")
print("- logistic_confusion_matrix.png")
print("- decision_tree_confusion_matrix.png")