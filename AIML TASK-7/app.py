from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os

# =====================================
# Load Model and Scaler
# =====================================

model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================
# Create FastAPI App
# =====================================

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict whether a student will Pass or Fail using a trained Machine Learning model.",
    version="1.0.0",
    contact={
        "name": "Muhammad Taha"
    }
)

# =====================================
# Input Schema
# =====================================

class StudentData(BaseModel):
    attendance: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    midterm_score: float = Field(..., ge=0, le=100)
    final_score: float = Field(..., ge=0, le=100)

# =====================================
# Home Endpoint
# =====================================

@app.get("/")
def home():
    return {
        "message": "Welcome to Student Performance Prediction API",
        "author": "Muhammad Taha",
        "internship": "HisabDo AI/ML Internship",
        "model": "Logistic Regression",
        "version": "1.0.0",
        "docs": "http://127.0.0.1:8000/docs"
    }

# =====================================
# Health Endpoint
# =====================================

@app.get("/health")
def health():
    return {
        "status": "API Running Successfully"
    }

# =====================================
# Project Info Endpoint
# =====================================

@app.get("/info")
def info():
    return {
        "project": "Student Performance Prediction",
        "author": "Muhammad Taha",
        "internship": "HisabDo AI/ML Internship",
        "day": "7",
        "algorithm": "Logistic Regression"
    }

# =====================================
# Prediction Endpoint
# =====================================

@app.post("/predict")
def predict(data: StudentData):

    features = np.array([[
        data.attendance,
        data.assignment_score,
        data.midterm_score,
        data.final_score
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    result = "Pass" if prediction == 1 else "Fail"
    confidence = round(max(probability) * 100, 2)

    if result == "Pass":
        message = "Congratulations! Student is likely to Pass."
    else:
        message = "Student is at risk of Failing."

    # Save Prediction History
    history = pd.DataFrame([{
        "Attendance": data.attendance,
        "Assignment Score": data.assignment_score,
        "Midterm Score": data.midterm_score,
        "Final Score": data.final_score,
        "Prediction": result,
        "Confidence": f"{confidence}%"
    }])

    if os.path.exists("prediction_history.csv"):
        history.to_csv(
            "prediction_history.csv",
            mode="a",
            header=False,
            index=False
        )
    else:
        history.to_csv(
            "prediction_history.csv",
            index=False
        )

    return {
        "prediction": result,
        "confidence": f"{confidence}%",
        "message": message,
        "model": "Logistic Regression",
        "api_version": "1.0.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }