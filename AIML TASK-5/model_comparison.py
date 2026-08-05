import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("student.csv")

print("First 5 Records")
print(df.head())

# ==========================
# Create Target Column
# ==========================
df["Pass"] = df["Final Score"].apply(lambda x: 1 if x >= 50 else 0)

print("\nPass/Fail Count")
print(df["Pass"].value_counts())

# ==========================
# Features & Target
# ==========================
X = df[["Attendance", "Assignment Score", "Midterm Score", "Final Score"]]
y = df["Pass"]

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data :", len(X_test))

# ==========================
# Logistic Regression
# ==========================
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ==========================
# Decision Tree
# ==========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# ==========================
# Evaluation
# ==========================
print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred, zero_division=0))
print("Recall   :", recall_score(y_test, lr_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, lr_pred, zero_division=0))

print("\n========== DECISION TREE ==========")
print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, zero_division=0))
print("Recall   :", recall_score(y_test, dt_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, dt_pred, zero_division=0))

# ==========================
# Confusion Matrices
# ==========================
cm_lr = confusion_matrix(y_test, lr_pred)
cm_dt = confusion_matrix(y_test, dt_pred)

print("\nLogistic Regression Confusion Matrix")
print(cm_lr)

print("\nDecision Tree Confusion Matrix")
print(cm_dt)

# ==========================
# Comparison Table
# ==========================
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred)
    ],
    "Precision": [
        precision_score(y_test, lr_pred, zero_division=0),
        precision_score(y_test, dt_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, lr_pred, zero_division=0),
        recall_score(y_test, dt_pred, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, lr_pred, zero_division=0),
        f1_score(y_test, dt_pred, zero_division=0)
    ]
})

print("\nModel Comparison")
print(comparison)

comparison.to_csv("comparison_results.csv", index=False)

# ==========================
# Accuracy Comparison Chart
# ==========================
plt.figure(figsize=(6,4))
plt.bar(comparison["Model"], comparison["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("comparison_chart.png")
plt.close()

# ==========================
# Confusion Matrix Charts
# ==========================
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("logistic_confusion_matrix.png")
plt.close()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.savefig("decision_tree_confusion_matrix.png")
plt.close()

print("\nAll files generated successfully!")
print("- comparison_results.csv")
print("- comparison_chart.png")
print("- logistic_confusion_matrix.png")
print("- decision_tree_confusion_matrix.png")