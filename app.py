import streamlit as st
import pandas as pd
import joblib
import numpy as np

#  1. LOAD YOUR SAVED MODEL AND SCALER
# Load the files you saved from Colab
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")


# Matched the exact spacing typos from the original CSV
original_columns = ['Year', 'Adult Mortality', 'infant deaths', 'Alcohol',
                    'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI',
                    'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
                    'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years', 
                    'thinness 5-9 years',    
                    'Income composition of resources',
                    'Schooling', 'Status_Developing']

# 2. CREATE THE WEB APP INTERFACE 
st.title('Cool Life Expectancy Predictor')
st.write("Enter the country's details to predict its life expectancy.")

# Create columns for layout
col1, col2 = st.columns(2)

# We use st.number_input for all the features
with col1:
    st.header("Socio-Economic Factors")
    year = st.number_input('Year', min_value=2000, max_value=2030, value=2014)
    gdp = st.number_input('GDP per capita (USD)', value=612.0)
    schooling = st.number_input('Schooling (Years)', value=10.0)
    income_comp = st.number_input('Income Composition of Resources', value=0.476, format="%.3f")
    status_dev = st.selectbox('Country Status', ['Developing', 'Developed'])

with col2:
    st.header("Health Factors")
    adult_mortality = st.number_input('Adult Mortality (per 1000)', value=271.0)
    hiv_aids = st.number_input('HIV/AIDS (deaths per 1000 births)', value=0.1, format="%.1f")
    bmi = st.number_input('BMI', value=18.6)
    alcohol = st.number_input('Alcohol (liters per capita)', value=0.01)
    # Add more inputs for other features as needed...

#  3. PREPARE THE INPUT FOR THE MODEL
if st.button('Predict Life Expectancy'):

    # 1. Convert Status to the 0/1 the model expects
    status_developing_val = 1 if status_dev == 'Developing' else 0

    # 2. Create the input dictionary
    #
    # 
    user_input = {
        'Year': year,
        'Adult Mortality': adult_mortality,
        'infant deaths': 64, # Using default values for simplicity here
        'Alcohol': alcohol,
        'percentage expenditure': 73.52, # Using default
        'Hepatitis B': 62.0, # Using default
        'Measles': 492, # Using default
        'BMI': bmi,
        'under-five deaths': 86, # Using default
        'Polio': 58.0, # Using default
        'Total expenditure': 8.18, # Using default
        'Diphtheria': 62.0, # Using default
        'HIV/AIDS': hiv_aids,
        'GDP': gdp,
        'Population': 327582.0, # Using default
        'thinness  1-19 years': 17.5, # <-- Double space key
        'thinness 5-9 years': 17.5,    # <-- Single space key
        'Income composition of resources': income_comp,
        'Schooling': schooling,
        'Status_Developing': status_developing_val
    }

    # 3. Convert to DataFrame in the correct order
    input_df = pd.DataFrame([user_input]).reindex(columns=original_columns)

    # 4. Scale the input using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # 5. Make the prediction
    prediction = model.predict(input_scaled)

    # 4. DISPLAY THE OUTPUT
    st.subheader(f'Predicted Life Expectancy: {prediction[0]:.2f} years')