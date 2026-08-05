# Student Performance Prediction - Model Comparison

## Overview

This project was completed for Day 5 of the HisabDo AI/ML Internship.

The objective was to compare two machine learning classification models for predicting whether a student will Pass or Fail.

## Models Used

- Logistic Regression
- Decision Tree Classifier

## Features Used

- Attendance
- Assignment Score
- Midterm Score
- Final Score

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |

## Conclusion

Both models achieved the same performance on this dataset.

The dataset is imbalanced because it contains many more Pass records than Fail records. A larger and more balanced dataset would provide a more realistic comparison between the models.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn