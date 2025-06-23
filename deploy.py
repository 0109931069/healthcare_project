import streamlit as st
import joblib
import pandas as pd
import numpy as np

Model = joblib.load("Third_Group.pkl")
Inputs = joblib.load("Inputs.pkl")

def Prediction(Age,Gender,Blood_Type,Medical_Condition,Medication,Admission_Type,Insurance_Provider,Billing_Amount):
    df = pd.DataFrame(columns=Inputs)
    df.at[0,"Gender"] = Gender
    df.at[0,"Blood Type"] = Blood_Type
    df.at[0,"Medical Condition"] = Medical_Condition
    df.at[0,"Medication"] = Medication
    df.at[0,"Admission Type"] = Admission_Type
    df.at[0,"Insurance Provider"] = Insurance_Provider
    df.at[0,"Billing Amount"] = Billing_Amount
    df.at[0,"Age"] = Age
    result = Model.predict(df)
    return result[0]

def Main():
    st.title("Health Care Prediction")
    Gender = st.selectbox("Gender",['Male', 'Female'])
    Blood_Type = st.selectbox("Blood_Type",['B-', 'AB-','A+','O-','A-','O+','B+','AB+'])
    Medical_Condition = st.selectbox("Medical Condition",[ 'Asthma',  'Obesity',  'Cancer',  'Arthritis','Hypertension','Diabetes'])
    Medication = st.selectbox("Medication",['Paracetamol', 'Ibuprofen','Penicillin','Aspirin','Lipitor'])
    Insurance_Provider = st.selectbox("Insurance Provider",['Cigna', 'Aetna', 'Medicare','UnitedHealthcare','Blue Cross'])
    Admission_Type = st.selectbox("Admission Type",['Emergency', 'Urgent','Elective'])
    Billing_Amount = st.slider("Billing_Amount",min_value=0.0 , max_value=51634.099835 , step=1.0,value = 10.0)
    Age = st.slider("Age",min_value=18.0 , max_value=85.0 , step=1.0,value = 10.0)
    
    
    
    if st.button("Predict"):
        result = Prediction(Age,Gender,Blood_Type,Medical_Condition,Medication,Admission_Type,Insurance_Provider,Billing_Amount)
        list_result = ["Abnormal" , "Normal",'Inconclusive']
        st.text(f"Your Health is {list_result[result]}")

    
    
Main()   