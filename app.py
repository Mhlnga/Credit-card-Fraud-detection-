import streamlit as st
import os
import io
import tempfile
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc


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
        "Model Evaluation",
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
    Search for a file in the project directory and its
    subdirectories, excluding Git and virtual environments.
    """
    for root, directories, files in os.walk(BASE_DIR):
        directories[:] = [
            directory for directory in directories
            if directory not in [
