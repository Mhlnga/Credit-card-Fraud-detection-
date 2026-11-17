
import streamlit as st
import os
import io
import tempfile
import pandas as pd

from pathlib import Path
from datetime import datetime


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. APPLICATION TITLE
# =====================================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An intelligent fraud detection application powered "
    "by a Convolutional Neural Network (CNN)."
)

st.markdown("---")


# =====================================================
# 3. NAVIGATION - DISPLAY BEFORE LOADING THE MODEL
# =====================================================

st.sidebar.title("💳 Fraud Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Single Transaction",
        "Batch Prediction",
        "Prediction History",
        "About"
    ],
    key="navigation"
)

st.sidebar.markdown("---")
st.sidebar.caption("CNN-Based Deep Learning Application")


# =====================================================
# 4. SESSION HISTORY
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_download_data" not in st.session_state:
    st.session_state.batch_download_data = None

if "batch_filename" not in st.session_state:
    st.session_state.batch_filename = None


# =====================================================
# 5. MODEL FILE LOCATIONS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent


def find_model_file(filename):

    """
    Search for a model file in the project directory
    and its subdirectories.

    Ignore Git and virtual environment directories.
    """

    for root, directories, files in os.walk(BASE_DIR):

        directories[:] = [
            directory for directory in directories
            if directory not in [
                ".git",
                ".venv",
                "venv",
                "__pycache__"
            ]
        ]

        for file in files:

            if file.lower() == filename.lower():

                return Path(root) / file

    return None


MODEL_PATH = find_model_file(
    "credit_card_fraud_cnn.keras"
)

SCALER_PATH = find_model_file(
    "credit_card_scaler.pkl"
)

FEATURES_PATH = find_model_file(
    "credit_card_features.pkl"
)


# =====================================================
# 6. MODEL FILE DIAGNOSTICS
# =====================================================

def show_model_diagnostics():

    st.subheader("Model File Diagnostics")

    st.write("Application directory:")

    st.code(str(BASE_DIR))

    st.write("Files detected by Streamlit:")

    detected_files = []

    for root, directories, files in os.walk(BASE_DIR):

        directories[:] = [
            directory for directory in directories
            if directory not in [
                ".git",
                ".venv",
                "venv",
                "__pycache__"
            ]
        ]

        for file in files:

            file_path = Path(root) / file

            relative_path = file_path.relative_to(BASE_DIR)

            detected_files.append(str(relative_path))

    if detected_files:

        st.code("\n".join(detected_files))

    else:

        st.warning("No project files were detected.")

    st.write("Required model files:")

    required_files = {
        "CNN Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Feature Names": FEATURES_PATH
    }

    for name, path in required_files.items():

        if path is not None and path.is_file():

            size_mb = path.stat().st_size / (1024 * 1024)

            st.success(
                f"{name}: Found ({size_mb:.2f} MB)"
            )

        else:

            st.error(f"{name}: Missing")


def is_git_lfs_pointer(file_path):

    """
    Check whether a file contains a Git LFS pointer
    instead of the actual model file.
    """

    try:

        with open(file_path, "rb") as file:

            first_line = file.readline(200)

        return first_line.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )

    except Exception:

        return False


# =====================================================
# 7. LOAD MODEL ONLY WHEN PREDICTIONS ARE REQUIRED
# =====================================================

@st.cache_resource
def load_model_files(model_path, scaler_path, features_path):

    import tensorflow as tf
    import joblib

    loaded_model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    loaded_scaler = joblib.load(
        scaler_path
    )

    loaded_features = joblib.load(
        features_path
    )

    return loaded_model, loaded_scaler, loaded_features


def get_model():

    required_files = {
        "CNN Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Feature Names": FEATURES_PATH
    }

    missing_files = [
        name
        for name, path in required_files.items()
        if path is None or not path.is_file()
    ]

    if missing_files:

        st.error(
            "Some required model files are missing:"
        )

        for filename in missing_files:

            st.write(f"- {filename}")

        show_model_diagnostics()

        st.warning(
            "Upload the missing model file to the "
            "GitHub repository used by Streamlit."
        )

        return None

    # Check whether the CNN file is only an LFS pointer
    if is_git_lfs_pointer(MODEL_PATH):

        st.error(
            "The CNN model is a Git LFS pointer, "
            "not the actual model file."
        )

        st.info(
            "Streamlit needs access to the actual "
            ".keras model file."
        )

        show_model_diagnostics()

        return None

    try:

        model, scaler, features = load_model_files(
            str(MODEL_PATH),
            str(SCALER_PATH),
            str(FEATURES_PATH)
        )

        if len(features) != 30:

            st.error(
                f"Expected 30 transaction features, "
                f"but found {len(features)}."
            )

            return None

        return model, scaler, features

    except Exception as error:

        st.error(
            "Unable to load the trained CNN model."
        )

        st.write("Technical error:")

        st.code(str(error))

        show_model_diagnostics()

        return None


# =====================================================
# 8. PREDICTION FUNCTION
# =====================================================

def predict_transactions(data, model, scaler, features):

    # Check required columns
    missing = [
        feature
        for feature in features
        if feature not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Arrange columns in training order
    data = data[features].copy()

    # Convert values to numbers
    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if data.isnull().values.any():

        raise ValueError(
            "The transaction data contains missing "
            "or non-numeric values."
        )

    # Scale the transaction data
    scaled_data = scaler.transform(data)

    # Reshape for CNN
    reshaped_data = scaled_data.reshape(
        scaled_data.shape[0],
        len(features),
        1
    )

    # Predict probabilities
    probabilities = model.predict(
        reshaped_data,
        verbose=0
    ).reshape(-1)

    # Classify transactions
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return predictions, probabilities


# =====================================================
# 9. HOME PAGE
# =====================================================

if page == "Home":

    st.header(
        "Welcome to the Fraud Detection System"
    )

    st.write(
        """
        This application uses a trained Convolutional Neural
        Network (CNN) to identify potentially fraudulent
        credit card transactions.

        It analyses transaction information and predicts
        whether transactions are normal or potentially fraudulent.
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

    total = sum(
        item["Total"] for item in history
    )

    fraud = sum(
        item["Fraud"] for item in history
    )

    normal = sum(
        item["Normal"] for item in history
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transactions Processed",
        f"{total:,}"
    )

    col2.metric(
        "Potential Fraud",
        f"{fraud:,}"
    )

    col3.metric(
        "Normal Transactions",
        f"{normal:,}"
    )


# =====================================================
# 10. SINGLE TRANSACTION PREDICTION
# =====================================================

elif page == "Single Transaction":

    st.header("Single Transaction Prediction")

    st.write(
        "Enter the transaction details below "
        "to check whether it may be fraudulent."
    )

    model_data = get_model()

    if model_data is not None:

        model, scaler, features = model_data

        with st.form("single_transaction_form"):

            st.subheader("Transaction Information")

            values = {}

            col1, col2 = st.columns(2)

            for index, feature in enumerate(features):

                with (
                    col1
                    if index % 2 == 0
                    else col2
                ):

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
                            value=0.0,
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

                predictions, probabilities = (
                    predict_transactions(
                        transaction,
                        model,
                        scaler,
                        features
                    )
                )

                prediction = int(
                    predictions[0]
                )

                probability = float(
                    probabilities[0]
                )

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

                st.session_state.prediction_history.append({

                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Source": "Single Transaction",

                    "Total": 1,

                    "Fraud": prediction,

                    "Normal": 1 - prediction

                })

            except Exception as error:

                st.error("Prediction failed.")

                st.code(str(error))


# =====================================================
# 11. BATCH PREDICTION
# =====================================================

elif page == "Batch Prediction":

    st.header("Batch Transaction Prediction")

    st.write(
        "Upload a CSV file containing credit card "
        "transactions for fraud detection."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="batch_csv_uploader"
    )

    if uploaded_file is not None:

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**File size:** "
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "Start Fraud Detection",
            use_container_width=True
        ):

            model_data = get_model()

            if model_data is not None:

                model, scaler, features = model_data

                temp_path = None

                try:

                    # Create temporary output file
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".csv",
                        delete=False,
                        encoding="utf-8",
                        newline=""
                    ) as temp_file:

                        temp_path = temp_file.name

                    total_transactions = 0

                    total_fraud = 0

                    total_normal = 0

                    preview = []

                    first_chunk = True

                    uploaded_file.seek(0)

                    progress = st.progress(0)

                    status = st.empty()

                    # Process CSV in batches
                    chunks = pd.read_csv(
                        uploaded_file,
                        chunksize=5000
                    )

                    for chunk in chunks:

                        predictions, probabilities = (
                            predict_transactions(
                                chunk,
                                model,
                                scaler,
                                features
                            )
                        )

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

                        total_transactions += len(chunk)

                        total_fraud += int(
                            (predictions == 1).sum()
                        )

                        total_normal += int(
                            (predictions == 0).sum()
                        )

                        # Keep up to 1000 rows for display
                        current_preview = sum(
                            len(item)
                            for item in preview
                        )

                        if current_preview < 1000:

                            remaining = (
                                1000 - current_preview
                            )

                            preview.append(
                                chunk.head(remaining)
                            )

                        # Save all results
                        chunk.to_csv(
                            temp_path,
                            mode="a",
                            index=False,
                            header=first_chunk
                        )

                        first_chunk = False

                        status.write(
                            f"Processed "
                            f"{total_transactions:,} transactions..."
                        )

                    # Read completed results
                    with open(
                        temp_path,
                        "rb"
                    ) as output_file:

                        download_data = output_file.read()

                    st.session_state.batch_results = (
                        pd.concat(
                            preview,
                            ignore_index=True
                        )
                        if preview
                        else pd.DataFrame()
                    )

                    st.session_state.batch_download_data = (
                        download_data
                    )

                    st.session_state.batch_filename = (
                        "fraud_detection_results.csv"
                    )

                    # Record batch history
                    st.session_state.prediction_history.append({

                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                        "Source": uploaded_file.name,

                        "Total": total_transactions,

                        "Fraud": total_fraud,

                        "Normal": total_normal

                    })

                    progress.progress(100)

                    status.success(
                        "All transactions processed successfully!"
                    )

                    st.success(
                        "Batch prediction completed."
                    )

                except Exception as error:

                    st.error(
                        "Batch prediction failed."
                    )

                    st.code(str(error))

                finally:

                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):

                        os.remove(temp_path)

    # Display previous batch results
    if st.session_state.batch_results is not None:

        st.markdown("---")

        st.subheader("Prediction Results")

        results = st.session_state.batch_results

        st.caption(
            "Showing up to the first 1,000 transactions."
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        if st.session_state.batch_download_data:

            st.download_button(
                "Download All Prediction Results",
                data=st.session_state.batch_download_data,
                file_name=st.session_state.batch_filename,
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# 12. PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.header("Prediction History")

    st.write(
        "View the transactions and CSV files processed "
        "during your current application session."
    )

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        total = int(
            history_df["Total"].sum()
        )

        fraud = int(
            history_df["Fraud"].sum()
        )

        normal = int(
            history_df["Normal"].sum()
        )

        st.subheader("History Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Processed",
            f"{total:,}"
        )

        col2.metric(
            "Potential Fraud",
            f"{fraud:,}"
        )

        col3.metric(
            "Normal Transactions",
            f"{normal:,}"
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
            "Make a prediction or upload a CSV file first."
        )


# =====================================================
# 13. ABOUT PAGE
# =====================================================

elif page == "About":

    st.header("About the Project")

    st.write(
        """
        The Credit Card Fraud Detection System is an
        Artificial Intelligence application developed
        using Deep Learning.

        It uses a Convolutional Neural Network (CNN)
        trained on credit card transaction data.

        The purpose of this system is to identify
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


# =====================================================
# 14. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection System | "
    "CNN-Based Deep Learning Application_By_Thembinkosi Emmanuel Mhlanga"
)
