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

I started with the Pima Indians Diabetes dataset and conducted thorough EDA. I noticed patterns such as abnormally high variance in insulin levels and invalid zeros in several columns (like BMI or blood pressure), which I cleaned by imputing median values. To capture non-obvious interactions, I engineered features like BMI × Age (to approximate metabolic stress over time), Glucose per Pregnancy (normalizing glucose by hormonal burden), and log transforms of skewed values like insulin. These features improved downstream model performance, especially for tree-based models.

I compared four models: Logistic Regression, Random Forest, XGBoost, and a PyTorch neural network (MLP). Logistic Regression was useful as a baseline—it’s interpretable and fast, but struggled with complex feature interactions and nonlinearities. Random Forest was better at capturing nonlinearity, but still showed some instability in probability calibration and was more memory-intensive. PyTorch gave me full control and flexibility, and I built a multi-layer network using ReLU and BCEWithLogitsLoss for stability. It performed reasonably (AUC ~0.83) but required hyperparameter tuning, careful learning rate control, and wasn’t interpretable or easy to deploy quickly.

Ultimately, I chose XGBoost for production. It consistently outperformed other models in both ROC AUC (~0.86–0.88) and stability. XGBoost’s built-in regularization (L1/L2), automatic handling of missing values, and ability to rank feature importance via SHAP values made it a robust and explainable choice. It generalizes well even without much tuning, and its compact model size makes deployment smooth. While deep learning gave me more architectural flexibility, it also added unnecessary complexity for a small tabular dataset. XGBoost offered the best balance between performance, interpretability, and practical deployment constraints.

To make the model accessible and demonstrate real-time inference, I built a Gradio app (https://huggingface.co/spaces/emily-wang2025/diabetes-risk-predictor). Users can enter health metrics and receive an instant diabetes risk score. The app wraps the XGBoost model using joblib for efficient inference, and the backend is designed to reflect real-world production use cases. The app code is located in app/gradio_app.py of the GitHub repo.

This project demonstrates my ability to apply thoughtful modeling decisions, engineer robust ML pipelines, and build interactive tools that turn models into products. I optimized every component not just for accuracy, but for clarity, stability, and user accessibility.

