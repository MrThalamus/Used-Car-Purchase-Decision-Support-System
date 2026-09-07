import streamlit as st

from utils.preprocessing import Preprocessor
from utils.predictor import Predictor

# -------------------------------------------------------
# Load Backend
# -------------------------------------------------------

preprocessor = Preprocessor()
predictor = Predictor()

# -------------------------------------------------------
# Prepare Dropdown Lists
# -------------------------------------------------------

if hasattr(preprocessor, "brand_list"):
    brand_list = sorted(preprocessor.brand_list)
else:
    brand_list = sorted(preprocessor.brand_frequency.keys())

if hasattr(preprocessor, "model_list"):
    model_list = sorted(preprocessor.model_list)
else:
    model_list = sorted(preprocessor.model_frequency.keys())

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Price Prediction",
    layout="wide"
)

st.title("Used Car Price Prediction")

st.write(
    "Provide the vehicle information below to predict its price category."
)

st.divider()

# -------------------------------------------------------
# Inputs
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
    "Brand",
    brand_list
    )

    if hasattr(preprocessor, "brand_model_map") and brand in preprocessor.brand_model_map:
      available_models = sorted(preprocessor.brand_model_map[brand])
    else:
        available_models = model_list

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

with col2:

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

    engine_capacity = st.number_input(
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

# -------------------------------------------------------
# Prediction Button
# -------------------------------------------------------

if st.button(
    "Predict Price Category",
    use_container_width=True
):

    try:

        brand_freq = preprocessor.brand_frequency.get(brand, 0.0)
        model_freq = preprocessor.model_frequency.get(model, 0.0)

        if brand_freq == 0.0 or model_freq == 0.0:
            st.warning(
                "The selected brand and/or model was not present in the training "
                "data, so the prediction for this vehicle may be less reliable."
            )

        X = preprocessor.transform(
            brand=brand,
            model=model,
            transmission=transmission,
            fuel_type=fuel,
            year=year,
            engine_capacity=engine_capacity,
            kilometers_run=kilometers,
        )

        result = predictor.predict(X)

        # -------------------------
        # Extra Calculations
        # -------------------------

        current_year = 2026
        car_age = max(current_year - year, 1)
        mileage_per_year = kilometers / car_age

        # -------------------------
        # Recommendation
        # -------------------------

        if result["prediction"] == "Luxury":
            recommendation = "Premium vehicle with high resale value and excellent features."
            condition = "Excellent"

        elif result["prediction"] == "High":
            recommendation = "High-value vehicle suitable for long-term ownership."
            condition = "Very Good"

        elif result["prediction"] == "Medium":
            recommendation = "Balanced option offering a good price-to-performance ratio."
            condition = "Good"

        else:
            recommendation = "Budget-friendly option. Inspect carefully before purchasing."
            condition = "Average"

        st.divider()

        st.header("Prediction Result")

        # -------------------------
        # Main Metrics
        # -------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Predicted Category",
                result["prediction"]
            )

        with c2:
            st.metric(
                "Confidence",
                f"{result['confidence']*100:.2f}%"
            )

        with c3:
            st.metric(
                "Vehicle Condition",
                condition
            )

        st.divider()

        # -------------------------
        # Vehicle Summary
        # -------------------------

        # st.subheader("Vehicle Summary")

        # c1, c2 = st.columns(2)

        # with c1:

        #     st.info(f"""
        #       **Brand:** {brand}

        #       **Model:** {model}

        #       **Fuel:** {fuel}

        #       **Transmission:** {transmission}
        #       """)

        # with c2:

        #                   st.info(f"""
        #       **Manufacturing Year:** {year}

        #       **Car Age:** {car_age} years

        #       **Engine Capacity:** {engine_capacity} cc

        #       **Mileage:** {kilometers:,} km
        #       """)

        st.divider()

        # -------------------------
        # DSS Insights
        # -------------------------

        st.subheader("Decision Support")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Mileage per Year",
                f"{mileage_per_year:,.0f} km"
            )

        with c2:

            st.metric(
                "Estimated Vehicle Age",
                f"{car_age} Years"
            )

        st.success(recommendation)

        st.divider()

        # -------------------------
        # Probability Distribution
        # -------------------------

        st.subheader("Prediction Probability")

        labels = predictor.label_encoder.classes_

        for label, prob in zip(labels, result["probability"]):

            st.write(f"**{label}**")

            st.progress(float(prob))

            st.caption(f"{prob*100:.2f}%")

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)