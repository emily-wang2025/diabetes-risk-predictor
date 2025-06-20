import gradio as gr
import numpy as np
import pandas as pd
import joblib

model = joblib.load("xgboost_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                     DiabetesPedigreeFunction, Age):
    BMI = 25
    bmi_age = BMI * Age
    glucose_per_preg = Glucose / (Pregnancies + 1)
    log_insulin = np.log1p(Insulin)
    log_bmi = np.log1p(BMI)

    df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        DiabetesPedigreeFunction, Age, bmi_age, glucose_per_preg,
        log_insulin, log_bmi
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "DiabetesPedigreeFunction", "Age", "BMI_Age", "GlucosePerPreg",
        "LogInsulin", "LogBMI"
    ])

    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]
    return f"Predicted diabetes risk: {prob:.2%}"

iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Risk Predictor",
    description="Estimate diabetes risk using a trained XGBoost model"
)

if __name__ == "__main__":
    iface.launch()