# Student Performance Prediction - Model Comparison

## Overview

This project was completed as part of the Day 5 task of the HisabDo AI/ML Internship.

The objective of this project is to compare two machine learning classification models to predict whether a student will Pass or Fail based on academic performance and attendance.

---

## Models Used

- Logistic Regression
- Decision Tree Classifier

---

## Features Used

- Attendance
- Assignment Score
- Midterm Score

---

## Target Variable

- Pass = 1
- Fail = 0

A student is considered **Pass** if:

- Attendance ≥ 75%
- Final Score ≥ 60

Otherwise, the student is classified as **Fail**.

---

## Machine Learning Process

1. Loaded the student dataset using Pandas.
2. Created the Pass/Fail target column.
3. Selected important features.
4. Split the dataset into training and testing sets.
5. Trained two classification models.
6. Evaluated both models using multiple performance metrics.
7. Compared the results and generated visualizations.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Conclusion

Both Logistic Regression and Decision Tree achieved the same performance on this dataset.

The dataset contains 30 student records with 21 Pass and 9 Fail cases. Because the dataset is relatively small and the classes are easy to separate, both models achieved 100% accuracy.

A larger and more diverse dataset would provide a more realistic comparison and better evaluate the strengths of different machine learning algorithms.

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Files Included

- student.csv
- model_comparison.py
- comparison_results.csv
- comparison_chart.png
- logistic_confusion_matrix.png
- decision_tree_confusion_matrix.png
- README.md
- requirements.txt