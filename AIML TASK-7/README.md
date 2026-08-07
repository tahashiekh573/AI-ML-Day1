# 🎓 Student Performance Prediction API

A Machine Learning-powered REST API built with **FastAPI** that predicts whether a student is likely to **Pass** or **Fail** based on academic performance.

This project was developed as part of the **HisabDo AI/ML Internship – Day 7 Assignment**.

---

# 📌 Project Overview

The API uses a trained **Logistic Regression** model to analyze student performance based on:

- Attendance
- Assignment Score
- Midterm Score
- Final Score

After receiving the input, the API:

- Validates the data
- Scales the features using StandardScaler
- Predicts Pass/Fail
- Returns prediction confidence
- Displays a user-friendly message
- Stores prediction history

---

# 🚀 Features

✅ FastAPI REST API

✅ Machine Learning Prediction

✅ Logistic Regression Model

✅ StandardScaler Preprocessing

✅ Prediction Confidence

✅ Input Validation (0–100)

✅ Prediction History Storage (CSV)

✅ Swagger UI Documentation

✅ Health Check Endpoint

✅ Project Information Endpoint

---

# 🛠 Technologies Used

- Python 3.x
- FastAPI
- Uvicorn
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Pydantic

---

# 📂 Project Structure

```text
AIML TASK-7/
│
├── screenshots/
│   ├── health.png
│   ├── info.png
│   ├── swagger_pass.png
│   ├── swagger_fail.png
│   ├── postman_pass.png
│   └── postman_fail.png
│
├── app.py
├── train_model.py
├── student.csv
├── student_model.pkl
├── scaler.pkl
├── prediction_history.csv
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone <repository-url>
cd AIML-TASK-7
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run the API

```bash
uvicorn app:app --reload
```

Server will start at

```
http://127.0.0.1:8000
```

---

# 📖 API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

# 📡 API Endpoints

## Home

```
GET /
```

Returns a welcome message.

---

## Health Check

```
GET /health
```

Response

```json
{
  "status": "API Running Successfully"
}
```

---

## Project Information

```
GET /info
```

Response

```json
{
  "project": "Student Performance Prediction",
  "author": "Muhammad Taha",
  "internship": "HisabDo AI/ML Internship",
  "day": "7",
  "algorithm": "Logistic Regression"
}
```

---

## Student Prediction

```
POST /predict
```

Request Body

```json
{
    "attendance": 90,
    "assignment_score": 85,
    "midterm_score": 80,
    "final_score": 88
}
```

---

## Successful Response

```json
{
    "prediction": "Pass",
    "confidence": "98.14%",
    "message": "Congratulations! Student is likely to Pass.",
    "model": "Logistic Regression",
    "api_version": "1.0.0",
    "timestamp": "2026-08-07 15:34:58"
}
```

---

## Fail Response

```json
{
    "prediction": "Fail",
    "confidence": "97.91%",
    "message": "Student is at risk of Failing.",
    "model": "Logistic Regression",
    "api_version": "1.0.0",
    "timestamp": "2026-08-07 15:35:33"
}
```

---

# 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Feature Selection
4. Train Logistic Regression Model
5. Save Model (.pkl)
6. Load Model into FastAPI
7. Accept User Input
8. Scale Features
9. Predict Pass/Fail
10. Return Confidence Score

---

# 📸 Screenshots

## Swagger UI

- Home Endpoint
- Health Endpoint
- Info Endpoint
- Pass Prediction
- Fail Prediction

## Postman

- Pass Prediction
- Fail Prediction

Screenshots are available inside the **screenshots/** folder.

---

# 📈 Future Improvements

- JWT Authentication
- Database Integration (MySQL/MongoDB)
- User Login System
- Model Retraining API
- Docker Deployment
- Cloud Deployment (Render/AWS/Azure)
- Streamlit Dashboard
- Multiple Machine Learning Algorithms
- Performance Comparison

---

# 👨‍💻 Author

**Muhammad Taha**

BS Computer Science

HisabDo AI/ML Internship

GitHub: https://github.com/tahashiekh573

---

# 📄 License

This project is developed for educational and internship purposes.

---

⭐ If you found this project helpful, don't forget to give it a star on GitHub!