# 🏦 Credit Risk Dashboard

A machine learning-powered credit risk scoring application built with Streamlit.  
It predicts loan default risk using the German Credit Dataset and provides real-time, explainable credit decisions.The project combines skills accross finance, statistics and coding.

## Features

- Interactive credit risk analytics dashboard  
- Real-time loan default prediction  
- Automated approval / rejection decision  
- Explainable AI using model coefficients  
- Portfolio risk insights and visualizations  
- Dataset explorer for analysis and auditing 

## Machine Learning Approach

- Model: Logistic Regression  
- Preprocessing: Scikit-learn Pipeline (ColumnTransformer)  
- Encoding: OneHotEncoding + StandardScaler  
- Output: Probability of default (0–1)  
- Evaluation: ROC-AUC, Confusion Matrix  

## Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- Plotly  

## Business Use Case

This project simulates a real-world banking credit scoring system used to evaluate loan applicants, support underwriting decisions, and manage portfolio risk.
streamlit run app.py
