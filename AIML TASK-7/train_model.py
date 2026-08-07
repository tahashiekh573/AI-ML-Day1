import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("student.csv")

# ==========================
# Create Target Column
# ==========================

df["Pass"] = (
    (df["Attendance"] >= 75) &
    (df["Final Score"] >= 60)
).astype(int)

# ==========================
# Features & Target
# ==========================

X = df[
    [
        "Attendance",
        "Assignment Score",
        "Midterm Score",
        "Final Score"
    ]
]

y = df["Pass"]

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# ==========================
# Train Model
# ==========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# ==========================
# Save Model & Scaler
# ==========================

joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")