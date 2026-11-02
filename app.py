import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("work_model.pkl")

st.title("Work Arrangement Predictor")
st.write("Predicting the work arrangement (Onsite, Hybrid or Remote) of an employee.")

# Inputs
age = st.number_input("Age", 18, 70, 30)
hours_per_week = st.number_input("Hours Per Week", 1, 100, 40)
balance = st.number_input("Work-Life Balance Score (1-5)", 1, 5, 3)
isolation = st.number_input("Social Isolation Score (1-5)", 1, 5, 3)

if st.button("Predict"):
    # Create DataFrame matching the training columns
    input_df = pd.DataFrame([[age, hours_per_week, balance, isolation]],
                            columns=["Age", "Hours_Per_Week", "Work_Life_Balance_Score", "Social_Isolation_Score"])

    # Prediction
    pred = model.predict(input_df)[0]
    st.success("Result: Predicted Work Arrangement is " + str(pred))
