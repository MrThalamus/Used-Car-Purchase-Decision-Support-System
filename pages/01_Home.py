import streamlit as st

st.set_page_config(
    page_title="Used Car Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: var(--text-color, inherit);
    }

    .sub-title {
        font-size: 22px;
        color: var(--text-color, inherit);
        opacity: 0.82;
    }

    .card {
        background-color: var(--secondary-background-color, rgba(127, 127, 127, 0.08));
        color: var(--text-color, inherit);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(127, 127, 127, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Used Car DSS")

st.sidebar.info(
    """
A Machine Learning based
Decision Support System for
Used Car Price Classification.
"""
)

st.sidebar.success("Navigate using the pages above.")

st.title("Home")

st.markdown(
    '<p class="main-title">Used Car Price Decision Support System</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="sub-title">Explainable Machine Learning for Used Vehicle Price Classification</p>',
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
<div class="card">

### Welcome

This application predicts the **price category** of a used vehicle using a
Machine Learning model trained on real-world used car listings.

The system performs:

- Feature Engineering
- Frequency Encoding
- One-Hot Encoding
- Feature Scaling
- Random Forest Classification
- Explainable AI (SHAP)

Use the navigation menu on the left to explore the application.

</div>
""",
    unsafe_allow_html=True,
)

st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Accuracy", "80.87%")

with col3:
    st.metric("Classes", "4")

st.divider()

st.subheader("Project Overview")

st.write(
"""
The system was developed using:

- Random Forest Classifier
- Feature Engineering
- Frequency Encoding
- One-Hot Encoding
- Standard Scaling
- Explainable AI (SHAP)
"""
)

st.divider()

st.subheader("Price Categories")

col1, col2 = st.columns(2)

with col1:
    st.success("Low")
    st.info("Medium")

with col2:
    st.warning("High")
    st.error("Luxury")

st.divider()

st.subheader("Navigation")

st.write(
"""
Use the sidebar to access:

- Price Prediction
- Model Performance
- Explainable AI
- Market Insights
- About
"""
)

st.divider()

st.subheader("Workflow")

st.write(
"""

User Input

->

Feature Engineering

->

Encoding & Scaling

->

Random Forest Model

->

Predicted Price Band

"""
)