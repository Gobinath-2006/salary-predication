import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    with open('salary prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# App Title and Description
st.title("💰 Salary Prediction App")
st.write("""
This app predicts the **Estimated Salary** based on **Years of Experience** using a Linear Regression model.
""")

# Sidebar or Main Input
st.subheader("Enter Details")
years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

# Prediction Button
if st.button("Predict Salary"):
    # Prepare input for model (expects 2D array or DataFrame with correct feature name)
    input_df = pd.DataFrame({'Years of Experience': [years_of_experience]})
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display Result
    st.success(f"### Predicted Salary: ${prediction[0]:,.2f}")

# Optional: Show Data Analysis
if st.checkbox("Show Sample Dataset"):
    df = pd.read_csv('Salary Data.csv')
    st.write(df.head(10))

# Optional: Visualize Trend
if st.checkbox("Show Salary Trend"):
    df = pd.read_csv('Salary Data.csv').dropna()
    st.line_chart(df.set_index('Years of Experience')['Salary'].sort_index())
