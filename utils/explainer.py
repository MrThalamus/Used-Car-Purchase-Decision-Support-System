from pathlib import Path
import warnings

import joblib
import shap
import pandas as pd
import numpy as np


class Explainer:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "models"

        # Filename is legacy; the file now holds a tuned XGBoost classifier, not a RandomForest.
        self.model = joblib.load(model_dir / "best_rf.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X):
        """Return a DataFrame with Feature, Value and SHAP columns for a single sample X.

        Handles different SHAP API versions and shapes (binary, multiclass, regression).
        Falls back gracefully and pads/trims SHAP values to match features if necessary.
        """
        # Ensure X is a DataFrame so feature names and values are available
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception:
                raise ValueError("Input X must be a pandas DataFrame or convertible to one")

        # Determine predicted class and class index when applicable
        predicted_class = None
        class_index = 0
        if hasattr(self.model, "predict"):
            try:
                predicted = self.model.predict(X)
                if hasattr(predicted, "__len__"):
                    predicted_class = predicted[0]
                else:
                    predicted_class = predicted
            except Exception:
                predicted_class = None

        if hasattr(self.model, "classes_") and predicted_class is not None:
            classes = list(self.model.classes_)
            if predicted_class in classes:
                class_index = classes.index(predicted_class)

        feature_names = list(X.columns)
        sample_values = X.iloc[0].values

        shap_vals = None

        # Try new shap Explanation object first
        try:
            explanation = self.explainer(X)
            feat_names_from_expl = getattr(explanation, "feature_names", None)
            if feat_names_from_expl is not None:
                feature_names = list(feat_names_from_expl)

            values = np.asarray(explanation.values)

            # Normalize shapes to a 1D shap vector for the requested class/sample
            if values.ndim == 2:
                # (n_samples, n_features)
                shap_vals = values[0]
            elif values.ndim == 3:
                # Possible shapes: (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
                n_classes = len(getattr(self.model, "classes_", [])) or values.shape[-1]
                if values.shape[-1] == n_classes:
                    shap_vals = values[0, :, class_index]
                elif values.shape[1] == n_classes:
                    shap_vals = values[0, class_index, :]
                else:
                    # Last resort: flatten and take first n_features
                    shap_vals = values.reshape(values.shape[1], -1).ravel()[: len(feature_names)]
            else:
                # Unexpected dims: flatten
                shap_vals = values.ravel()[: len(feature_names)]

        except Exception:
            # Older SHAP versions or other failures: try shap_values API
            try:
                sv = self.explainer.shap_values(X)

                # shap_values may be a list (multiclass) or array
                if isinstance(sv, list):
                    # Each element has shape (n_samples, n_features)
                    if len(sv) == len(getattr(self.model, "classes_", sv)):
                        shap_vals = np.asarray(sv[class_index])[0]
                    else:
                        # Unexpected list length, try first element
                        shap_vals = np.asarray(sv[0])[0]
                else:
                    # Array-like: (n_samples, n_features)
                    shap_vals = np.asarray(sv)[0]

            except Exception as e:
                warnings.warn(f"Failed to compute SHAP values: {e}. Returning zeros.")
                shap_vals = np.zeros(len(feature_names))

        # Ensure shap_vals is a 1D array of length matching feature_names
        shap_vals = np.asarray(shap_vals, dtype=float)
        if shap_vals.size < len(feature_names):
            # pad with zeros
            pad = np.zeros(len(feature_names) - shap_vals.size)
            shap_vals = np.concatenate([shap_vals.ravel(), pad])
        elif shap_vals.size > len(feature_names):
            shap_vals = shap_vals.ravel()[: len(feature_names)]

        # Build DataFrame
        df = pd.DataFrame({
            "Feature": feature_names,
            "Value": sample_values,
            "SHAP": shap_vals
        })

        df["Impact"] = df["SHAP"].abs()
        return df.sort_values("Impact", ascending=False).reset_index(drop=True)