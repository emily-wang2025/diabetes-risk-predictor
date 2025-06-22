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
One of the projects I’m most proud of is a machine learning system I designed and deployed to predict diabetes risk using health screening data. This project involved the full ML engineering pipeline—from raw data to interactive web deployment. The complete code is here: https://github.com/emily-wang2025/diabetes-risk-predictor and the live web app is running at https://huggingface.co/spaces/emiwang/gradio .


This was a solo project where I was responsible for everything — data cleaning, EDA, feature engineering, model training and comparison, hyperparameter tuning, evaluation, and final deployment through a Gradio interface hosted on Hugging Face Spaces. The project reflects my ability to work across the full ML lifecycle — from raw data to a live web app.

I started with the Pima Indians Diabetes dataset and conducted thorough EDA. I noticed patterns such as abnormally high variance in insulin levels and invalid zeros in several columns (like BMI or blood pressure), which I cleaned by imputing median values. To capture non-obvious interactions, I engineered features like BMI × Age (to approximate metabolic stress over time), Glucose per Pregnancy (normalizing glucose by hormonal burden), and log transforms of skewed values like insulin. These features improved downstream model performance, especially for tree-based models.

After data cleaning (handling missing values and normalizing skewed distributions), I focused on feature engineering, which proved essential in boosting model performance and generalization. For example:
-	I created BMI_Age, a product of BMI and age, to reflect how obesity in older individuals may indicate a higher or different risk profile compared to younger individuals.
-	I added GlucosePerPreg (Glucose / (Pregnancies + 1)), which normalizes glucose relative to pregnancy count — capturing a physiological nuance in the dataset.
-	I applied log transformations (e.g., LogInsulin, LogBMI) to reduce skew and stabilize variance for models sensitive to outliers.
These features significantly improved performance — especially in models like Random Forest and XGBoost, where feature importance confirmed their predictive value.
I trained and evaluated several models, including Logistic Regression, Random Forest, XGBoost (default and tuned), and a PyTorch-based neural network. I used RandomizedSearchCV to tune XGBoost’s hyperparameters efficiently across a large parameter space.
Model Comparison and Trade-offs
I trained and compared four models:
-	Logistic Regression: Strong interpretability and solid baseline ROC AUC (0.8233), but weaker precision and recall.
-	Random Forest: Best overall balance — highest ROC AUC (0.8319), strong recall, robust to overfitting, and interpretable via feature importance.
-	XGBoost (tuned with RandomizedSearchCV): Performed well in terms of precision and recall, and benefited from regularization, but its ROC AUC (0.8156) was slightly lower than Random Forest. While more powerful on large structured datasets, here it added complexity without a performance gain.
-	PyTorch Neural Network: Achieved the highest ROC AUC (~0.8452), but I ultimately excluded it due to small dataset size, limited interpretability, and lack of validation techniques like dropout or early stopping.
I selected Random Forest as the final model for deployment, based on its consistent performance and transparency. To make the model accessible and demonstrate real-time inference, I built a Gradio app (https://huggingface.co/spaces/emily-wang2025/diabetes-risk-predictor). Users can enter health metrics and receive an instant diabetes risk score. The app wraps the Random Forest model using joblib for efficient inference, and the backend is designed to reflect real-world production use cases. The app code is located in app/gradio_app.py of the GitHub repo.

