import pandas as pd

# Load Dataset
df = pd.read_csv("student.csv")

# Create Pass/Fail column
df["Pass"] = df["Final Score"].apply(lambda x: 1 if x >= 50 else 0)

# Display first 5 rows
print(df.head())

# Show Pass/Fail count
print("\nPass/Fail Count")
print(df["Pass"].value_counts())

from sklearn.model_selection import train_test_split

# Features (Input)
X = df[["Attendance", "Assignment Score", "Midterm Score", "Final Score"]]

# Target (Output)
y = df["Pass"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))

from sklearn.linear_model import LogisticRegression

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make Predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Pass/Fail Distribution
plt.figure(figsize=(5,4))
df["Pass"].value_counts().plot(kind="bar")
plt.title("Pass / Fail Distribution")
plt.xlabel("Pass=1  Fail=0")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("pass_fail_distribution.png")
plt.close()

# Confusion Matrix Chart
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("\nCharts saved successfully!")