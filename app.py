
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "to identify potentially fraudulent credit card transactions."
)

MODEL_PATH = "credit_card_fraud_cnn.keras"
SCALER_PATH = "credit_card_scaler.pkl"
FEATURES_PATH = "credit_card_features.pkl"

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

# Check that all required files exist
required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error("The following required files are missing from GitHub:")
    for file in missing_files:
        st.write(f"- {file}")
    st.info("Upload these files to the same folder as app.py.")
    st.stop()

try:
    model, scaler, features = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

st.success("CNN model loaded successfully.")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Single Transaction", "Batch Prediction", "About"]
)

def predict_transactions(data):
    data = data[features].copy()
    scaled = scaler.transform(data)
    reshaped = scaled.reshape(scaled.shape[0], 30, 1)
    probabilities = model.predict(reshaped, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "Use this system to assess individual transactions or upload "
        "a CSV file containing multiple transactions."
    )
    st.info(
        "Select Single Transaction to enter transaction details, "
        "or Batch Prediction to upload a CSV file."
    )

elif page == "Single Transaction":
    st.subheader("Check an Individual Transaction")

    with st.form("transaction_form"):
        values = {}

        st.write("Enter the transaction details:")

        for feature in features:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"input_{feature}"
            )

        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        transaction = pd.DataFrame([values], columns=features)
        predictions, probabilities = predict_transactions(transaction)

        probability = float(probabilities[0])

        st.metric("Fraud Probability", f"{probability * 100:.2f}%")

        if predictions[0] == 1:
            st.error("⚠️ Potentially Fraudulent Transaction")
        else:
            st.success("✅ Transaction Classified as Normal")

elif page == "Batch Prediction":
    st.subheader("Upload Transactions for Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            missing = [col for col in features if col not in data.columns]

            if missing:
                st.error(f"Your CSV is missing these columns: {missing}")
            else:
                predictions, probabilities = predict_transactions(data)

                results = data.copy()
                results["Prediction"] = predictions
                results["Fraud Probability"] = probabilities
                results["Result"] = results["Prediction"].map({
                    0: "Normal",
                    1: "Potential Fraud"
                })

                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)

                fraud_count = int((predictions == 1).sum())
                normal_count = int((predictions == 0).sum())

                col1, col2 = st.columns(2)
                col1.metric("Normal Transactions", normal_count)
                col2.metric("Potential Fraud", fraud_count)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "About":
    st.subheader("About This Project")
    st.write(
        "The Credit Card Fraud Detection System uses a "
        "Convolutional Neural Network (CNN) trained on transaction "
        "features to classify transactions as normal or potentially fraudulent."
    )
    st.warning(
        "This is an educational project. Predictions should not be "
        "used as the sole basis for real financial decisions."
    )
