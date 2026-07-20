import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Used Car Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

home_page = st.Page("pages/01_Home.py", title="Home", default=True)
price_prediction_page = st.Page("pages/02_Price_prediction.py", title="Price Prediction")
model_performance_page = st.Page("pages/03_Model_Performance.py", title="Model Performance")
explainable_ai_page = st.Page("pages/04_Explainable_AI.py", title="Explainable AI")
market_insights_page = st.Page("pages/05_Market_Insights.py", title="Market Insights")
about_page = st.Page("pages/06_About.py", title="About")

navigation = st.navigation(
    [
        home_page,
        price_prediction_page,
        model_performance_page,
        explainable_ai_page,
        market_insights_page,
        about_page,
    ]
)

navigation.run()