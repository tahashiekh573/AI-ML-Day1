# Student Performance Prediction - Model Optimization

## Overview

This project was completed as part of the **Day 6** task of the **HisabDo AI/ML Internship**.

The objective of this project is to improve the performance of a machine learning model using **Feature Engineering**, **Feature Scaling**, and **Hyperparameter Tuning**. The project predicts whether a student will **Pass** or **Fail** based on attendance and academic performance.

The optimized model is compared with the previous Logistic Regression model using multiple evaluation metrics.

---

## Objectives

- Perform Data Cleaning
- Apply Feature Engineering
- Select Important Features
- Scale Numerical Features
- Split Dataset into Training and Testing Sets
- Optimize Logistic Regression using GridSearchCV
- Compare Previous and Tuned Models
- Evaluate Model Performance
- Visualize Results

---

## Dataset

The dataset contains **30 student records** with the following information:

- Student Name
- Age
- Gender
- Course
- Attendance
- Assignment Score
- Midterm Score
- Final Score

---

## Target Variable

A new target column **Pass** was created.

**Pass = 1** if:

- Attendance ≥ 75%
- Final Score ≥ 60

Otherwise:

**Fail = 0**

---

## Feature Engineering

The following preprocessing steps were performed:

- Checked for missing values
- Removed duplicate records
- Created Pass/Fail target column
- Selected important numerical features
- Applied Feature Scaling using StandardScaler

---

## Features Used

The following features were selected for model training:

- Attendance
- Assignment Score
- Midterm Score

---

## Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

The split was performed using:

- train_test_split()
- stratify=y
- random_state=42

---

## Models Used

### Previous Model

- Logistic Regression
- Default Parameters

### Tuned Model

- Logistic Regression
- Optimized using GridSearchCV

---

## Hyperparameter Tuning

GridSearchCV was used to find the best model parameters.

### Parameter Grid

```python
{
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}
```

### Best Parameters

```
C = 0.01
Solver = liblinear
```

---

## Evaluation Metrics

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Previous Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tuned Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Visualizations

The project generates the following visualizations:

- Model Accuracy Comparison Chart
- Confusion Matrix
- ROC Curve
- Feature Importance Chart

---

## Generated Files

The project automatically creates the following files:

- student.csv
- model_optimization.py
- comparison_results.csv
- predictions.csv
- comparison_chart.png
- confusion_matrix.png
- roc_curve.png
- feature_importance.png
- README.md
- requirements.txt
---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Workflow

1. Load the dataset.
2. Check for missing values and duplicates.
3. Create the Pass/Fail target column.
4. Select important features.
5. Split the dataset into training and testing sets.
6. Scale the features using StandardScaler.
7. Train the baseline Logistic Regression model.
8. Optimize the model using GridSearchCV.
9. Evaluate both models using multiple metrics.
10. Generate comparison tables and graphs.
11. Save prediction results and visualizations.

---

## Conclusion

This project demonstrates the complete workflow of improving a machine learning model through **Feature Engineering** and **Hyperparameter Tuning**.

The optimized Logistic Regression model achieved the same performance as the previous model because the dataset is relatively small and the Pass/Fail classes are easily separable. Hyperparameter tuning identified the best model configuration, while feature scaling improved data consistency.

For real-world applications, using a larger and more diverse dataset would provide a better comparison between models and produce more reliable performance evaluation.

---

## Author

**Muhammad Taha**

AI/ML Internship – HisabDo

**Day 6 Project**