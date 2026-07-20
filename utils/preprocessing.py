from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


class Preprocessor:
    """
    Handles all feature engineering before prediction.
    """
    print("Loading preprocessing.py...")
    def __init__(self):

        print("Preprocessor initialized")

        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_DIR = BASE_DIR / "models"

        self.brand_model_map = joblib.load(MODEL_DIR / "brand_model_map.pkl")

        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

        self.brand_frequency = joblib.load(MODEL_DIR / "brand_frequency.pkl")
        self.model_frequency = joblib.load(MODEL_DIR / "model_frequency.pkl")

        self.brand_list = joblib.load(MODEL_DIR / "brand_list.pkl")
        self.model_list = joblib.load(MODEL_DIR / "model_list.pkl")

        print("brand_list loaded:", len(self.brand_list))
        print("model_list loaded:", len(self.model_list))

    def transform(
        self,
        brand,
        model,
        transmission,
        fuel_type,
        year,
        engine_capacity,
        kilometers_run,
    ):

        # -------------------------
        # Feature Engineering
        # -------------------------

        current_year = datetime.now().year

        car_age = max(current_year - int(year), 1)

        mileage_per_year = float(kilometers_run) / car_age

        brand_freq = self.brand_frequency.get(brand, 0.0)
        model_freq = self.model_frequency.get(model, 0.0)

        # -------------------------
        # Create Empty DataFrame
        # IMPORTANT:
        # use 0.0 instead of 0
        # -------------------------

        data = pd.DataFrame(
            [[0.0] * len(self.feature_names)],
            columns=self.feature_names,
        )

        # -------------------------
        # Numerical Features
        # -------------------------

        data.loc[0, "energy_capacity"] = float(engine_capacity)
        data.loc[0, "kilometers_run"] = float(kilometers_run)
        data.loc[0, "car_age"] = float(car_age)
        data.loc[0, "mileage_per_year"] = float(mileage_per_year)
        data.loc[0, "brand_frequency"] = float(brand_freq)
        data.loc[0, "model_frequency"] = float(model_freq)

        # -------------------------
        # Fuel Encoding
        # -------------------------

        fuel_column = f"fuel_{fuel_type.lower()}"

        if fuel_column in data.columns:
            data.loc[0, fuel_column] = 1.0

        # -------------------------
        # Brand Encoding
        # -------------------------

        brand_column = f"brand_{brand}"

        if brand_column in data.columns:
            data.loc[0, brand_column] = 1.0

        # -------------------------
        # Model Encoding
        # -------------------------

        model_column = f"model_{model}"

        if model_column in data.columns:
            data.loc[0, model_column] = 1.0

        # -------------------------
        # Transmission Encoding
        # -------------------------

        transmission_column = f"transmission_{transmission}"

        if transmission_column in data.columns:
            data.loc[0, transmission_column] = 1.0

        # Keep correct feature order
        data = data[self.feature_names]

        # Scale ONLY the first 13 columns
        data.iloc[:, :13] = self.scaler.transform(data.iloc[:, :13])

        return data