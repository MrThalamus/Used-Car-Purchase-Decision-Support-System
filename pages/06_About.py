import streamlit as st

st.set_page_config(page_title="About", layout="centered")

st.title("About This Project")

st.markdown(
    """
    **Used Car Price Decision Support System** is a lightweight Streamlit application that demonstrates
    an end-to-end machine learning workflow for classifying used vehicle price categories and providing
    model explainability using SHAP.

    This app includes:

    - Interactive price prediction with explainable AI (SHAP) for individual vehicle inputs.
    - Market Insights with distributions, trends, and top brands/models.
    - Model Performance dashboard with classification metrics, confusion matrix, ROC curves, and feature importance.

    """
)

st.divider()

st.header("Data")
st.markdown(
    """
    The dataset used in this project is a curated collection of used vehicle listings. Each row represents a single
    listing and includes features such as brand, model, transmission, fuel type, manufacturing year, engine capacity,
    kilometers driven, and the assigned price band.

    The application expects the CSV to be present at data/used_car.csv relative to the project root. If you want to
    update the dataset, replace that file and then refresh the app.
    """
)

st.header("Model & Explainability")
st.markdown(
    """
    The predictive model is a tuned XGBoost classifier (serialized under models/best_rf.pkl — the filename is
    legacy from an earlier Random Forest version). Brand and model are represented by their frequency in the
    training data rather than one-hot columns; the Preprocessor utility handles this frequency encoding along
    with scaling of the numeric features. SHAP (TreeExplainer) is used to produce per-sample explanations shown
    on the Explainable AI page.
    """
)

st.header("How to Use")
st.markdown(
    """
    1. Use the left-side navigation to select a page (Home, Price Prediction, Explainable AI, Market Insights, Model Performance).
    2. On Explainable AI: provide vehicle attributes and click "Explain Prediction" to see SHAP-based explanations and natural-language summaries.
    3. On Market Insights: filter the dataset by Brand / Transmission / Year range and explore charts and downloadable filtered data.
    4. On Model Performance: adjust the sample fraction and seed to evaluate model metrics on a random subset (or the full dataset).
    """
)

st.header("Developer & License")
st.markdown(
    """
    - Author: Research / Data Mining team
    - Repository: Local project workspace
    - License: MIT-like (please adapt as appropriate for your project)
    """
)

st.header("Contact & Feedback")
st.markdown(
    """
    For questions, bug reports, or feature requests, please open an issue in the project repository or contact the maintainer.
    If you'd like help extending the app (adding new visualizations, model types, or deployment), send a message here and
    the assistant can implement the changes.
    """
)

st.info("Version: 1.0 — Last updated: 2026-07-20")