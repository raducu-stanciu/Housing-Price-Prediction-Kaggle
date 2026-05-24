import streamlit as st
import pandas as pd
import joblib

def add_features(X):
    X_copy = X.copy()
    X_copy['TotalSF'] = X_copy['1stFlrSF'] + X_copy['2ndFlrSF'] + X_copy['TotalBsmtSF'] 
    X_copy['HouseAge'] = X_copy['YrSold'] - X_copy['YearBuilt']
    X_copy['TotalBath'] = X_copy['FullBath'] + (0.5 * X_copy['HalfBath']) + X_copy['BsmtFullBath']
    return X_copy


#  Load the monolithic pipeline
model = joblib.load('house_price_model.pkl')

st.title("🏠 House Price Predictor")


with st.expander("ℹ️ List of abbreviations"):
    st.markdown("""
    **Quality:**
    * **Ex:** Excellent | **Gd:** Good | **TA:** Typical / Average | **Fa:** Fair
    
    **Zoning:**
    * **RL:** Residential Low Density
    * **RM:** Residential Medium Density
    * **FV:** Floating Village Residential
    * **RH:** Residential High Density
    * **C (all):** Commercial
    """)

#  UI for the Top 10 Features
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Inputs contributing to TotalSF & Size
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
        gr_liv_area = st.number_input("Living Area (sqft)", value=1500)
        first_flr = st.number_input("1st Floor Area (sqft)", value=1000)
        total_bsmt = st.number_input("Total Basement Area (sqft)", value=1000)
        garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2)

    with col2:
        # Inputs for Quality & Features
        exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"], index=2)
        kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], index=2)
        full_bath = st.number_input("Full Bathrooms", value=2)
        central_air = st.radio("Central Air Conditioning", ["Y", "N"], index=0)
        ms_zoning = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0)

    submit = st.form_submit_button("Predict Price")

if submit:
    
    #  Created a dictionary with ALL 76 features initialized to a safe default value.
    # This ensures the model receives the exact shape it expects.
    input_data = {col: 0 for col in model.feature_names_in_}

    
    input_data.update({
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        '1stFlrSF': first_flr,
        'TotalBsmtSF': total_bsmt,
        'GarageCars': garage_cars,
        'ExterQual': exter_qual,
        'KitchenQual': kitchen_qual,
        'FullBath': full_bath,
        'CentralAir': central_air,
        'MSZoning': ms_zoning,
        # Mandatory ingredients for add_features even if not in Top 10
        '2ndFlrSF': 0, 
        'HalfBath': 0,
        'BsmtFullBath': 0,
        'YearBuilt': 2005,
        'YrSold': 2026
    })



    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predictcion
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Sale Price: ${prediction:,.2f}")