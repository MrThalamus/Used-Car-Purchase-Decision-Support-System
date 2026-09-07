import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import Preprocessor
from utils.predictor import Predictor
from utils.explainer import Explainer

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Explainable AI",
    layout="wide"
)

# ==========================================================
# Load Backend
# ==========================================================

preprocessor = Preprocessor()
predictor = Predictor()
explainer = Explainer()

# ==========================================================
# Title
# ==========================================================

st.title("Explainable AI")
st.markdown(
    """
Understand **why** the AI predicted a particular **Used Car Price Category**
using **SHAP (SHapley Additive exPlanations)**.
"""
)

st.divider()

# ==========================================================
# Input Section
# ==========================================================

left, right = st.columns(2)

with left:

    brand = st.selectbox(
        "Brand",
        sorted(preprocessor.brand_list)
    )

    if brand in preprocessor.brand_model_map:
        available_models = sorted(preprocessor.brand_model_map[brand])
    else:
        available_models = sorted(preprocessor.model_list)

    model = st.selectbox(
        "Model",
        available_models
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Automatic",
            "Manual"
        ]
    )

with right:

    fuel = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "Hybrid",
            "Electric",
            "CNG",
            "LPG",
            "Octane"
        ]
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2018
    )

    engine = st.number_input(
        "Engine Capacity (cc)",
        min_value=600,
        max_value=7000,
        value=1500,
        step=100
    )

    kilometers = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=60000,
        step=1000
    )

st.write("")

# ==========================================================
# Explain Button
# ==========================================================

if st.button("Explain Prediction", use_container_width=True):

    brand_freq = preprocessor.brand_frequency.get(brand, 0.0)
    model_freq = preprocessor.model_frequency.get(model, 0.0)

    if brand_freq == 0.0 or model_freq == 0.0:
        st.warning(
            "The selected brand and/or model was not present in the training "
            "data, so this explanation may be less reliable."
        )

    with st.spinner("Generating explanation..."):

        X = preprocessor.transform(
            brand=brand,
            model=model,
            transmission=transmission,
            fuel_type=fuel,
            year=year,
            engine_capacity=engine,
            kilometers_run=kilometers
        )

        prediction = predictor.predict(X)

        explanation = explainer.explain(X)

    # ======================================================
    # Prediction Result
    # ======================================================

    st.success(
        f"### Predicted Price Category : **{prediction['prediction']}**"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction Confidence",
            f"{prediction['confidence']*100:.2f}%"
        )

    with col2:

        st.metric(
            "Predicted Class",
            prediction["prediction"]
        )

    st.divider()

    # ======================================================
    # Probability Chart
    # ======================================================

    st.subheader("Prediction Probabilities")

    labels = predictor.label_encoder.classes_

    probability_df = pd.DataFrame({
        "Category": labels,
        "Probability": prediction["probability"]
    })

    fig = px.bar(
        probability_df,
        x="Category",
        y="Probability",
        color="Probability",
        text="Probability",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_layout(
        height=420,
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # Top Features
    # ======================================================

    st.subheader("Top 5 Most Influential Features")

    top = explanation.head(5).copy()

    fig2 = px.bar(
        top,
        x="Impact",
        y="Feature",
        orientation="h",
        color="SHAP",
        color_continuous_scale="RdBu"
    )

    fig2.update_layout(
        height=600,
        yaxis=dict(autorange="reversed"),
        coloraxis_colorbar_title="SHAP"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # Detailed Table
    # ======================================================

    st.subheader("Detailed SHAP Values")

    st.dataframe(
        explanation,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ======================================================
    # Natural Language Explanation
    # ======================================================

    st.subheader("AI Interpretation")

    positive = explanation.sort_values(
        "SHAP",
        ascending=False
    ).head(5)

    negative = explanation.sort_values(
        "SHAP"
    ).head(5)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Features Increasing the Prediction")

        for _, row in positive.iterrows():

            st.success(
                f"""
**{row['Feature']}**

Value: **{row['Value']:.4f}**

Contribution: **+{row['SHAP']:.4f}**
"""
            )

    with col2:

        st.markdown("### Features Decreasing the Prediction")

        for _, row in negative.iterrows():

            st.error(
                f"""
**{row['Feature']}**

Value: **{row['Value']:.4f}**

Contribution: **{row['SHAP']:.4f}**
"""
            )

    st.divider()

    # ======================================================
    # Summary
    # ======================================================

    st.info(
        """
### Interpretation

• Positive SHAP values pushed the prediction **towards** the predicted class.

• Negative SHAP values pushed the prediction **away** from the predicted class.

• Larger absolute SHAP values indicate a stronger influence on the model's decision.

This explanation is generated using **SHAP (TreeExplainer)** for the trained XGBoost model.
"""
    )