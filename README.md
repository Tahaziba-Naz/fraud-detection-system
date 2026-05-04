# Fraud Detection System

This project focuses on detecting fraudulent transactions using machine learning.

## Problem
The dataset is highly imbalanced, with very few fraud cases compared to normal transactions. This makes it difficult for models to correctly identify fraud.

## What I did
- Loaded and explored the dataset using Pandas
- Cleaned the data and removed unnecessary columns
- Observed class imbalance in the dataset
- Used SMOTE to balance the dataset
- Trained models like Logistic Regression and Random Forest
- Evaluated the model using precision, recall, and F1-score

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Streamlit

## Result
The model was able to detect fraudulent transactions better by focusing on recall instead of accuracy.

## Demo
<img width="1918" height="963" alt="Screenshot 2026-05-04 134939" src="https://github.com/user-attachments/assets/5e82d217-12b4-4408-abea-dba1608431f2" />

<img width="1919" height="970" alt="image" src="https://github.com/user-attachments/assets/878afd86-45b8-4f2f-a643-9a06bd116adc" />

## Dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud

## What I learned
- Handling imbalanced datasets
- Importance of recall in fraud detection
- How machine learning can be applied to real-world problems
