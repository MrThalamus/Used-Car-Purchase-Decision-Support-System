import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ------------------------
# Models
# ------------------------

def load_model():
    return joblib.load(ROOT / "models" / "best_rf.pkl")


def load_scaler():
    return joblib.load(ROOT / "models" / "scaler.pkl")


def load_label_encoder():
    return joblib.load(ROOT / "models" / "label_encoder.pkl")


def load_feature_names():
    return joblib.load(ROOT / "models" / "feature_names.pkl")


def load_kmeans():
    return joblib.load(ROOT / "models" / "kmeans.pkl")


# ------------------------
# Dictionaries
# ------------------------

def load_brand_frequency():
    return joblib.load(ROOT / "models" / "brand_frequency.pkl")


def load_model_frequency():
    return joblib.load(ROOT / "models" / "model_frequency.pkl")


# ------------------------
# Lists
# ------------------------

def load_brand_list():
    return joblib.load(ROOT / "models" / "brand_list.pkl")


def load_model_list():
    return joblib.load(ROOT / "models" / "model_list.pkl")


# ------------------------
# Dataset
# ------------------------

def load_dataset():
    return pd.read_csv(ROOT / "data" / "used_car.csv")