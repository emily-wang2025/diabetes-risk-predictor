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

The app allows users to input health metrics and receive a real-time risk prediction. It uses the trained Random Forest model saved via joblib for fast inference and is structured to mirror real-world production interfaces. The app code is available in the app/gradio_app.py file of the GitHub repository.

I recently designed and deployed a machine learning system to predict diabetes risk using clinical health screening data. I built this project entirely on my own — from exploratory data analysis and feature engineering to model training, tuning, and live deployment through a Gradio web app.

- Code: https://github.com/emily-wang2025/diabetes-risk-predictor/blob/main/Diabetes_Risk_Prediction.ipynb
- App: https://huggingface.co/spaces/emiwang/gradio

## Exploratory Data Analysis, Feature Engineering and Preprocessing

This project showcases both my modeling and engineering capabilities. I started with the Pima Indians Diabetes dataset and applied a clean separation between train/test sets to prevent leakage. I explored the data visually and statistically to understand missing values, skew, and outliers. I noticed several clinical columns (e.g., insulin, BMI, skin thickness) had invalid zero values. These were imputed with medians after confirming skew via histograms. I also log-transformed skewed features like insulin to stabilize variance and added new engineered features such as:
-	BMI_Age (BMI × Age): to approximate metabolic load over time. Aging amplifies the effects of obesity on metabolic health. This feature models that compounding effect and improved the signal especially for Random Forest and XGBoost.
-	GlucosePerPreg (glucose / pregnancies + 1): to normalize blood sugar relative to hormonal strain. Glucose levels in pregnant individuals are affected by hormonal changes. This feature normalizes glucose by hormonal load, giving a clearer picture of dysregulation.
-	LogInsulin, LogBMI: to reduce skew and improve generalization for linear and tree-based models. Insulin and BMI had long-tailed distributions. Log transforms reduced skew and stabilized variance — helping linear models like Logistic Regression and improving consistency across models.

All features were scaled using StandardScaler, and categorical/continuous distinctions were respected.

## Modeling Choices, Evaluation, and Iteration
I trained and compared four models: Logistic Regression, Random Forest, XGBoost, and a PyTorch neural network. I used precision, recall, and ROC AUC as evaluation metrics, and trained models using 80/20 splits to monitor overfitting. For XGBoost, I implemented RandomizedSearchCV for hyperparameter tuning to control complexity. This included tuning max_depth, subsample, and learning_rate, which improved precision and recall but still underperformed Random Forest in AUC. I also added a neural net built from scratch in PyTorch using nn.Sequential, trained with BCELoss and Adam.
- Logistic Regression
--	Pros: Fast, interpretable, no risk of overfitting
--	Cons: Assumes linearity; underperformed on recall
-- Result: AUC = 0.8233 — solid baseline, but not good enough for deployment
- Random Forest
•	Pros: High AUC (0.8319), robust to noise and outliers, great generalization
•	Cons: Less precise than XGBoost; not ideal for extrapolation
•	Why I chose it: It offered the best overall balance of performance, consistency, and explainability (via feature_importances_). I confirmed generalization using AUC and recall on both train and test sets.
3. XGBoost (tuned)
•	Pros: Great precision and recall after tuning, handles skew and imbalance well
•	Cons: Slightly lower AUC (0.8156) than Random Forest; added complexity didn’t result in clear gain
•	Notes: Used RandomizedSearchCV to tune max_depth, subsample, learning_rate, and n_estimators. Useful to learn tuning workflows even though RF performed better.
4. PyTorch Neural Network
•	Pros: Highest AUC (0.8452), captured nonlinearities
•	Cons: Prone to overfitting on small datasets, lower interpretability, no dropout or validation tracking
•	Result: Impressive performance, but not deployable due to instability and lower trust


Despite the neural net achieving the highest ROC AUC (~0.8452), I ultimately deployed the Random Forest (AUC = 0.8319) because it had the best balance of generalization, interpretability, and consistency. It was also easier to explain through built-in feature importances. I confirmed no overfitting by plotting AUC on both training and test sets and through precision-recall curves.

## Deployment & Engineering
To make the model interactive and usable by non-technical users, I wrapped the Random Forest model using joblib and built a real-time web app using Gradio, hosted on Hugging Face Spaces. The app allows users to enter their health stats and receive a probability score. The model code and deployment script are modular and production-ready (gradio_app.py), showing my ability to translate ML models into real interfaces — a core skill for this LinkedIn role.
![image](https://github.com/user-attachments/assets/99c9c65b-0aaa-4b8b-80b9-9b6a1e1174ef)


