import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import  LabelEncoder

# Load saved components
model = load_model('house_price.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('df_cat_encoded.pkl', 'rb') as f:
    label_encoders = pickle.load(f)  # Dictionary of LabelEncoders per categorical column


train_df = pd.read_csv('train.csv')

# Define feature columns
numerical_cols = [
    "PID", "Lot Frontage", "Lot Area", "Overall Qual", "Year Built", "Year Remod/Add",
    "Mas Vnr Area", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF",
    "Gr Liv Area", "Bsmt Full Bath", "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr",
    "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt", "Garage Cars", "Garage Area",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch"
]

categorical_cols = [
    "MS Zoning", "Street", "Lot Shape", "Land Contour", "Utilities", "Lot Config", "Land Slope",
    "Neighborhood", "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style",
    "Roof Matl", "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Exter Qual", "Exter Cond",
    "Foundation", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Heating", "Heating QC", "Central Air", "Electrical", "Kitchen Qual", "Functional",
    "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond", "Paved Drive",
    "Sale Type", "Sale Condition"
]

# Streamlit UI
st.title("🏠 Ames Housing Sale Price Predictor")

user_input = {}

st.header("Enter House Features")

# Input fields
for col in numerical_cols:
    user_input[col] = st.number_input(col, step=1.0)

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()   # <-- Correct way to create LabelEncoder instance
    le.fit(train_df[col])  
    label_encoders[col] = le
    user_input[col] = st.selectbox(col, label_encoders[col].classes_)

# Predict button
if st.button("Predict Sale Price"):
    try:
        # Separate and preprocess inputs
        num_data = pd.DataFrame([{
            col: user_input[col] for col in numerical_cols
        }])

        cat_data = pd.DataFrame([{
            col: label_encoders[col].transform([user_input[col]])[0] for col in categorical_cols
        }], index=[0])

        # Combine and scale
        full_data = pd.concat([num_data, cat_data], axis=1)
        full_data_scaled = scaler.transform(full_data)

        # Predict
        prediction = model.predict(full_data_scaled)
        st.success(f" Predicted Sale Price of home: ${prediction[0][0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
