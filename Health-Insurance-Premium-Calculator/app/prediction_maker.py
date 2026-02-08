import streamlit as st
import numpy as np
import pandas as pd
import joblib
scaler_old = joblib.load("artifacts/scaler_old.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
model_old = joblib.load("artifacts/model_old.joblib")
model_young = joblib.load("artifacts/model_young.joblib")

def calculate_normalised_risk_score(input_df):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = input_df.lower().split(" & ")
    total_risk_score = sum([risk_scores.get(disease,0) for disease in diseases])
    max_score = 14
    min_score = 0
    normalised_risk_score = (total_risk_score - min_score)/(max_score - min_score)
    return normalised_risk_score

def preprocessing(input_df):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalised_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    for key, value in input_df.items():
        if key == 'age' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    df["normalised_risk_score"] = calculate_normalised_risk_score(input_df["Medical History"])

    if input_df['Age'] >=25:
        scaler = scaler_young
    else:
        scaler = scaler_old
    cols_to_scale = scaler['cols_to_scale']
    print("printing cols to scale")
    print(cols_to_scale)
    scaler = scaler['scaler']
    df['income_level'] = None  # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    preprocessed_df = preprocessing(input_dict)
    print(scaler_young)
    print(scaler_old)
    print(preprocessed_df.columns)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(preprocessed_df)
    else:
        preprocessed_df.drop('genetical_risk', axis='columns', inplace=True)
        prediction = model_old.predict(preprocessed_df)
    return int(prediction[0])