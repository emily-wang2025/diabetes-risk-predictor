# Diabetes Risk Predictor

This project predicts diabetes risk using real-world health data and machine learning models, including a PyTorch neural network.

## Features
- Perform EDA and handle missing/invalid values
- Build a strong preprocessing pipeline with feature engineering
- Compare multiple models (LogReg, RF, XGBoost, tuned XGBoost, Neural Net)
- Visualize model performance (ROC, AUC, feature importance)
- Explain predictions with SHAP
- Deploy a live web app with Gradio

## Explaination
One of the machine learning projects I enjoyed creating is a diabetes risk prediction system that I designed, developed, and deployed end-to-end using real-world health screening data. This project involved the full ML engineering pipeline—from data ingestion and preprocessing to modeling, evaluation, and live deployment.
Full code: https://github.com/emily-wang2025/diabetes-risk-predictor
Live app: https://huggingface.co/spaces/emiwang/gradio

As a solo project, I owned every part of the pipeline: data cleaning, exploratory data analysis (EDA), feature engineering, model training and selection, hyperparameter tuning, and deployment through a Gradio interface hosted on Hugging Face Spaces. The project demonstrates my ability to execute across the full machine learning lifecycle and translate models into interactive, production-ready tools.

I started with the Pima Indians Diabetes dataset and conducted thorough EDA. I identified issues such as invalid zero values in several columns (e.g., insulin, BMI, blood pressure), which I addressed by imputing medians. I also discovered strong skew and variance in features like insulin and glucose. To address these, I engineered several new features to capture more nuanced clinical signals:

- BMI_Age: The product of BMI and age, designed to approximate cumulative metabolic stress and highlight risk in older individuals with obesity.

- GlucosePerPreg: Glucose level divided by (pregnancies + 1), a domain-informed feature to reflect glucose regulation relative to hormonal changes.

- LogInsulin, LogBMI: Log transformations to reduce skew, stabilize variance, and improve model robustness, particularly for linear models.

These engineered features significantly improved model performance, especially for tree-based models like Random Forest and XGBoost. Feature importance rankings confirmed their contribution.

## Model Comparison and Trade-offs

I trained and evaluated multiple models: Logistic Regression, Random Forest, XGBoost (both default and tuned via RandomizedSearchCV), and a PyTorch-based neural network. Each model came with trade-offs:

- Logistic Regression offered high interpretability and a strong baseline AUC (0.8233), but underperformed on recall and precision.

- Random Forest achieved the best overall performance with the highest ROC AUC (0.8319), strong recall, and robustness to overfitting. It also provided interpretability through built-in feature importance, which made it ideal for deployment.

- Tuned XGBoost performed well on recall and precision and benefited from regularization, but its ROC AUC (0.8156) was slightly lower, and the added complexity did not result in measurable gains for this dataset.

- Neural Network (PyTorch) achieved the highest AUC (~0.8452) but was excluded from deployment due to limited interpretability, small dataset size, and lack of regularization features like dropout or early stopping.

Given the trade-offs, I selected Random Forest as the final model due to its balance of accuracy, robustness, and interpretability.

To make the system accessible, I built and deployed a Gradio web application:
https://huggingface.co/spaces/emiwang/gradio


The app allows users to input health metrics and receive a real-time risk prediction. It uses the trained Random Forest model saved via joblib for fast inference and is structured to mirror real-world production interfaces. The app code is available in the app/gradio_app.py file of the GitHub repository.


