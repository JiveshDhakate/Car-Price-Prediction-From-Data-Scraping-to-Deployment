import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# --- Load Model ---
model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.pkl")
model = joblib.load(model_path)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter your car details below to estimate its market value.</p>", unsafe_allow_html=True)
st.write("---")

# --- Input Layout ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", ['Audi', 'BMW', 'Mercedes-Benz', 'Volkswagen'])
        classification = st.selectbox("Classification", ['SUV', 'Hatchback', 'Saloon', 'Estate', 'Coupe', 'Convertible'])
        transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'Petrol Hybrid', 'Diesel Hybrid', 'Electric'])
        location = st.selectbox("Sale Location", [
            'Dublin', 'Cork', 'Galway', 'Limerick', 'Waterford', 'Kilkenny', 'Mayo', 'Donegal',
            'Louth', 'Wexford', 'Meath', 'Kildare', 'Westmeath', 'Tipperary', 'Laois', 'Longford',
            'Sligo', 'Leitrim', 'Monaghan', 'Clare', 'Offaly', 'Cavan', 'Carlow', 'Roscommon', 'Wicklow', 'Kerry'
        ])
        year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, value=2018)
        mileage = st.number_input("Mileage (in km)", min_value=0, max_value=300000, value=50000)

    submit_button = st.form_submit_button("💸 Predict Price")

# --- Prediction Logic ---
if submit_button:
    input_df = pd.DataFrame([{
        'Year': year,
        'Mileage': mileage,
        'Brand_' + brand: 1,
        'Classification_' + classification: 1,
        'Transmission_' + transmission: 1,
        'Fuel Type_' + fuel_type: 1,
        'Sale Location_' + location: 1
    }])

    # Align with training features
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
    if model_features is not None:
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

    # Predict
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: €{predicted_price:,.2f}")
    st.balloons()

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>Created with by <b>Jivesh Dhakate</b></p>",
    unsafe_allow_html=True
)
