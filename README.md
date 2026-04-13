# Customer Churn Forecaster (End-to-End Machine Learning Project)

This project predicts whether a customer is likely to churn (leave the service) based on their demographic and service usage information. It is built as an end-to-end machine learning pipeline and deployed using Streamlit.

## Live App
[Streamlit App](https://customerchurnforecaster.streamlit.app/)

## Overview
Customer churn is a major problem for subscription-based businesses. The goal of this project is to build a predictive model that identifies customers who are likely to churn so that retention strategies can be applied proactively.

## Dataset
The dataset includes customer information such as:

- Gender, SeniorCitizen, Partner, Dependents  
- Tenure  
- Services (Phone, Internet, Streaming, etc.)  
- Contract type  
- Payment method  
- Monthly and Total charges  

Target variable:
- `Churn` (Yes/No)

## Machine Learning Workflow

### 1. Data Preprocessing
- Handled categorical and numerical variables
- Scaled and Encoded features using preprocessing pipeline

### 2. Class Imbalance Handling
- Used SMOTE during training to handle imbalanced dataset

### 3. Models Used
- Decision Tree Classifier  
- Random Forest Classifier  
- HistGradientBoosting Classifier  

### 4. Model Selection
Best model selected based on **F1-score**

## Model Performance

### Baseline Models:
- Decision Tree F1-score: 0.5068  
- Random Forest F1-score: 0.5859  
- HistGradientBoosting F1-score: 0.5932

Best model: HistGradientBoosting

### Final Hyperparameter Tuned Model (using GridSearchCV):
- Accuracy: 0.78  
- Weighted F1-score: 0.79  
- Recall (Churn Class): 0.73  

## Evaluation Metrics

### Confusion Matrix:
```
[[1235  311]
[ 155  412]]
```

### Classification Report:
| Class     | Precision | Recall | F1-score | Support |
|-----------|----------|--------|----------|---------|
| No Churn  | 0.89     | 0.80   | 0.84     | 1546    |
| Churn     | 0.57     | 0.73   | 0.64     | 567     |

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

## Deployment

The model is deployed using **Streamlit Cloud**.

Users can input customer details and get real time churn predictions.

## Project Structure
```
├── app.py                # Streamlit web app
├── model.pkl             # Trained ML pipeline
├── churn.ipynb           # Model training & EDA
├── requirements.txt      # Dependencies
└── README.md
```
