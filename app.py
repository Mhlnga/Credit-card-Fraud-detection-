
import os
import io
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. APPLICATION TITLE
# =========================================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An intelligent fraud detection application powered by "
    "a Convolutional Neural Network (CNN)."
)

st.markdown("---")


# =========================================================
# 3. MODEL FILE PATHS
# =========================================================

MODEL_PATH = "credit_card_fraud_cnn.keras"
SCALER_PATH = "credit_card_scaler.pkl"
FEATURES_PATH = "credit_card_features.pkl"


# =========================================================
# 4. INITIALISE SESSION HISTORY
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_download_data" not in st.session_state:
    st.session_state.batch_download_data = None

if "batch_filename" not in st.session_state:
    st.session_state.batch_filename = None


# =========================================================
# 5. CHECK REQUIRED MODEL FILES
# =========================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH
]

missing_files = [
    filename for filename in required_files
    if not os.path.exists(filename)
]

if missing_files:
    st.error("Required model files are missing.")

    for filename in missing_files:
        st.write(f"- {filename}")

    st.info(
        "Upload all three model files into the same "
        "GitHub folder as app.py."
    )

    st.stop()


# =========================================================
# 6. LOAD THE TRAINED CNN MODEL
# =========================================================

@st.cache_resource
def load_model_files():

    loaded_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    loaded_scaler = joblib.load(SCALER_PATH)

    loaded_features = joblib.load(FEATURES_PATH)

    return loaded_model, loaded_scaler, loaded_features


try:

    model, scaler, features = load_model_files()

except Exception as error:

    st.error("Unable to load the trained CNN model.")

    st.code(str(error))

    st.stop()


if len(features) != 30:

    st.error(
        f"The model expects 30 features, but "
        f"{len(features)} were loaded."
    )

    st.stop()


st.success("CNN model loaded successfully!")


# =========================================================
# 7. PREDICTION FUNCTION
# =========================================================

def predict_transactions(data):

    # Arrange columns in the same order used during training
    data = data[features].copy()

    # Convert all values to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isnull().values.any():

        raise ValueError(
            "The uploaded data contains missing or "
            "non-numeric feature values."
        )

    # Apply the original training scaler
    scaled_data = scaler.transform(data)

    # Reshape data for CNN input
    reshaped_data = scaled_data.reshape(
        scaled_data.shape[0],
        30,
        1
    )

    # Predict fraud probabilities
    probabilities = model.predict(
        reshaped_data,
        verbose=0
    ).reshape(-1)

    # Classify transactions
    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities


# =========================================================
# 8. NAVIGATION MENU
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Single Transaction",
        "Batch Prediction",
        "Prediction History",
        "About"
    ]
)


# =========================================================
# 9. HOME PAGE
# =========================================================

if page == "Home":

    st.header("Welcome to the Fraud Detection System")

    st.write(
        """
        This application uses a trained Convolutional Neural
        Network to identify potentially fraudulent credit
        card transactions.

        The system analyses transaction information and
        predicts whether a transaction is normal or potentially
        fraudulent.
        """
    )

    st.subheader("System Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **Single Transaction Prediction**

            Enter transaction details and receive an
            individual fraud prediction.
            """
        )

    with col2:

        st.info(
            """
            **Batch Prediction**

            Upload a CSV file containing multiple transactions
            and receive predictions for each transaction.
            """
        )

    st.subheader("Current Session Statistics")

    history = st.session_state.prediction_history

    total = sum(item["Total"] for item in history)

    fraud = sum(item["Fraud"] for item in history)

    normal = sum(item["Normal"] for item in history)

    col1, col2, col3 = st.columns(3)

    col1.metric("Transactions Processed", f"{total:,}")

    col2.metric("Potential Fraud", f"{fraud:,}")

    col3.metric("Normal Transactions", f"{normal:,}")


# =========================================================
# 10. SINGLE TRANSACTION PREDICTION
# =========================================================

elif page == "Single Transaction":

    st.header("Single Transaction Prediction")

    st.write(
        "Enter the transaction information below "
        "to check whether it may be fraudulent."
    )

    with st.form("single_transaction_form"):

        st.subheader("Transaction Information")

        values = {}

        col1, col2 = st.columns(2)

        for index, feature in enumerate(features):

            with (col1 if index % 2 == 0 else col2):

                default_value = 0.0

                if feature == "Amount":
                    values[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        format="%.6f"
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        format="%.6f"
                    )

        submitted = st.form_submit_button(
            "Check Transaction",
            use_container_width=True
        )

    if submitted:

        try:

            transaction = pd.DataFrame(
                [values],
                columns=features
            )

            predictions, probabilities = predict_transactions(
                transaction
            )

            prediction = int(predictions[0])

            probability = float(probabilities[0])

            fraud_count = prediction
            normal_count = 1 - prediction

            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            if prediction == 1:

                st.error(
                    "⚠️ Potentially Fraudulent Transaction"
                )

            else:

                st.success(
                    "✅ Transaction Classified as Normal"
                )

            # Add transaction to history
            st.session_state.prediction_history.append({

                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Source": "Single Transaction",

                "Total": 1,

                "Fraud": fraud_count,

                "Normal": normal_count

            })

        except Exception as error:

            st.error(
                f"Prediction error: {str(error)}"
            )


# =========================================================
# 11. BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.header("Batch Transaction Prediction")

    st.write(
        "Upload a CSV file containing credit card "
        "transactions to detect potentially fraudulent activity."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="batch_csv_uploader"
    )

    if uploaded_file is not None:

        st.write(
            f"**Selected file:** {uploaded_file.name}"
        )

        st.write(
            f"**File size:** "
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "Start Fraud Detection",
            use_container_width=True
        ):

            temp_path = None

            try:

                # Reset file position
                uploaded_file.seek(0)

                # Process file in batches
                chunks = pd.read_csv(
                    uploaded_file,
                    chunksize=5000
                )

                total_transactions = 0

                total_fraud = 0

                total_normal = 0

                preview = []

                csv_buffer = io.StringIO()

                first_chunk = True

                progress_bar = st.progress(0)

                status_text = st.empty()

                # Create temporary output file
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    delete=False,
                    encoding="utf-8",
                    newline=""
                ) as temp_file:

                    temp_path = temp_file.name

                # Process each CSV batch
                for chunk in chunks:

                    # Check required columns
                    missing_columns = [
                        feature for feature in features
                        if feature not in chunk.columns
                    ]

                    if missing_columns:

                        raise ValueError(
                            "Your CSV file is missing these "
                            f"columns: {missing_columns}"
                        )

                    # Predict transactions
                    predictions, probabilities = (
                        predict_transactions(chunk)
                    )

                    # Add prediction information
                    chunk = chunk.copy()

                    chunk["Prediction"] = predictions

                    chunk["Fraud Probability"] = probabilities

                    chunk["Result"] = pd.Series(
                        predictions,
                        index=chunk.index
                    ).map({
                        0: "Normal",
                        1: "Potential Fraud"
                    })

                    # Update statistics
                    total_transactions += len(chunk)

                    total_fraud += int(
                        (predictions == 1).sum()
                    )

                    total_normal += int(
                        (predictions == 0).sum()
                    )

                    # Save first 1000 rows for display
                    if sum(len(item) for item in preview) < 1000:

                        remaining = 1000 - sum(
                            len(item) for item in preview
                        )

                        preview.append(
                            chunk.head(remaining)
                        )

                    # Write results to temporary CSV
                    chunk.to_csv(
                        temp_path,
                        mode="a",
                        index=False,
                        header=first_chunk
                    )

                    first_chunk = False

                    status_text.write(
                        f"Processed {total_transactions:,} transactions..."
                    )

                # Read completed output
                with open(
                    temp_path,
                    "rb"
                ) as output_file:

                    download_data = output_file.read()

                # Save results in session
                st.session_state.batch_results = pd.concat(
                    preview,
                    ignore_index=True
                ) if preview else pd.DataFrame()

                st.session_state.batch_download_data = download_data

                st.session_state.batch_filename = (
                    "fraud_detection_results.csv"
                )

                # Add batch to history
                st.session_state.prediction_history.append({

                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Source": uploaded_file.name,

                    "Total": total_transactions,

                    "Fraud": total_fraud,

                    "Normal": total_normal

                })

                progress_bar.progress(100)

                status_text.success(
                    "Batch prediction completed successfully!"
                )

                st.success(
                    "All transactions have been processed."
                )

            except Exception as error:

                st.error(
                    f"Batch prediction failed: {str(error)}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)

    # Display results after processing
    if st.session_state.batch_results is not None:

        st.markdown("---")

        st.subheader("Prediction Results")

        history = st.session_state.prediction_history

        if history:

            latest = history[-1]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                f"{latest['Total']:,}"
            )

            col2.metric(
                "Normal Transactions",
                f"{latest['Normal']:,}"
            )

            col3.metric(
                "Potential Fraud",
                f"{latest['Fraud']:,}"
            )

        st.caption(
            "The table below displays up to the first "
            "1,000 processed transactions."
        )

        st.dataframe(
            st.session_state.batch_results,
            use_container_width=True
        )

        if st.session_state.batch_download_data:

            st.download_button(
                label="Download All Prediction Results",
                data=st.session_state.batch_download_data,
                file_name=st.session_state.batch_filename,
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# 12. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.header("Prediction History")

    st.write(
        "View the transactions and CSV files processed "
        "during your current application session."
    )

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        st.subheader("History Summary")

        total_transactions = history_df["Total"].sum()

        total_fraud = history_df["Fraud"].sum()

        total_normal = history_df["Normal"].sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Processed",
            f"{total_transactions:,}"
        )

        col2.metric(
            "Potential Fraud",
            f"{total_fraud:,}"
        )

        col3.metric(
            "Normal Transactions",
            f"{total_normal:,}"
        )

        st.subheader("Processing History")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        history_csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []

            st.session_state.batch_results = None

            st.session_state.batch_download_data = None

            st.session_state.batch_filename = None

            st.rerun()

    else:

        st.info(
            "No prediction history is available yet. "
            "Make a single prediction or upload a CSV file "
            "to start building your history."
        )


# =========================================================
# 13. ABOUT PAGE
# =========================================================

elif page == "About":

    st.header("About the Project")

    st.write(
        """
        The Credit Card Fraud Detection System is an
        artificial intelligence application developed
        using Deep Learning.

        It uses a Convolutional Neural Network (CNN)
        trained on credit card transaction data.

        The purpose of the system is to identify
        potentially fraudulent transactions and
        distinguish them from normal transactions.
        """
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - TensorFlow
        - Keras
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        """
    )

    st.warning(
        "This application is an educational project. "
        "Its predictions should not be used as the sole "
        "basis for real financial decisions."
    )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection System | "
    "CNN-Based Deep Learning Application"
)
