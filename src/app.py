import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💰 Loan Default Prediction App")

path = os.path.join(os.path.dirname(__file__), "cleaned_loan_df.pkl")  # Path to the cleaned data
df = joblib.load(path)  # Cleaned data for reference
encoder = joblib.load("src/encoder.pkl")    # Encoder for categorical features
scaler = joblib.load("src/scaler.pkl")      # Scaler for numerical features
model_path = os.path.join(os.path.dirname(__file__), "final_pipeline_with_meta.pkl") # Path to the model
model = joblib.load(model_path)              # Load the model pipeline
pipeline = model["model"]

features_config = joblib.load("src/features_config.pkl") # Load features configuration

# Categorical and numerical features
categorical_cols = features_config['categorical_features']
numerical_cols = features_config['numeric_features']
ordinal_features = features_config['ordinal_feature']
binary_features = features_config['binary_features']

# st.title("Loan Default Prediction App")

# --- Take inputs from user ---
user_input = {}

st.header("Enter applicant information:")

for col in numerical_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, value=0.0)

for col in categorical_cols:
    options = df[col].unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

for col in ordinal_features:
    options = df[col].unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

for col in binary_features: 
    options = df[col].unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

    # convert input to DataFrame
input_df = pd.DataFrame([user_input])
st.dataframe(input_df, hide_index=True)

# prediction
if st.button("Predict Loan Default"):
       # Make prediction
    prediction = pipeline.predict(input_df)
    prob = pipeline.predict_proba(input_df)[:, 1][0]  # Probability of default
    # st.write(prediction)
    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ Likely to Default with probability {prob:.2f}")
    else:
        st.success(f"✅ Unlikely to Default with probability {1 - prob:.2f}")
    

