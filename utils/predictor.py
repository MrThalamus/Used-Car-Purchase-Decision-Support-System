import joblib
from pathlib import Path


class Predictor:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_DIR = BASE_DIR / "models"

        # Filename is legacy; the file now holds a tuned XGBoost classifier, not a RandomForest.
        self.model = joblib.load(MODEL_DIR / "best_rf.pkl")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    def predict(self, X):

        prediction = self.model.predict(X)[0]

        probability = self.model.predict_proba(X)[0]

        predicted_label = self.label_encoder.inverse_transform([prediction])[0]

        confidence = float(probability.max())

        return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probability": probability.tolist()
    }