import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)

from utils import loader
from utils.preprocessing import Preprocessor

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance")
st.markdown("Assess model accuracy, class-level metrics, confusion matrix, ROC curves, and feature importance.")
st.divider()

# ----------------------------------------------------------
# Load data and model
# ----------------------------------------------------------
@st.cache_data
def get_data():
    df = loader.load_dataset()
    # normalize column names
    if "year_of_manufacture" in df.columns and "year" not in df.columns:
        df["year"] = df["year_of_manufacture"]
    return df

@st.cache_resource
def get_objects():
    preprocessor = Preprocessor()
    model = loader.load_model()
    label_encoder = loader.load_label_encoder()
    feature_names = loader.load_feature_names()
    return preprocessor, model, label_encoder, feature_names


df = get_data()
preprocessor, model, label_encoder, feature_names = get_objects()

# ----------------------------------------------------------
# Sidebar options
# ----------------------------------------------------------
st.sidebar.header("Evaluation Options")

sample_frac = st.sidebar.slider("Sample fraction for evaluation", 0.05, 1.0, 0.5, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
show_sample = st.sidebar.checkbox("Show sample predictions table", value=True)

# Prepare evaluation sample
sampled = df.sample(frac=sample_frac, random_state=int(random_state)).reset_index(drop=True)

# Helper to extract single fuel (first listed)
def _normalize_fuel(f):
    if pd.isna(f):
        return "Unknown"
    if isinstance(f, str) and "," in f:
        return f.split(",")[0].strip()
    return str(f).strip()

# Build X matrix for the model using Preprocessor.transform row-wise
@st.cache_data
def build_feature_matrix(rows):
    X_list = []
    for _, r in rows.iterrows():
        fuel = _normalize_fuel(r.get("fuel_type", ""))
        year = r.get("year") or r.get("year_of_manufacture") or r.get("car_age")
        try:
            year_val = int(year)
        except Exception:
            year_val = 2018

        x = preprocessor.transform(
            brand=r.get("brand", "Other"),
            model=r.get("model", "Other"),
            transmission=r.get("transmission", "Automatic"),
            fuel_type=fuel,
            year=year_val,
            engine_capacity=r.get("energy_capacity", r.get("engine_capacity", 1500)),
            kilometers_run=r.get("kilometers_run", r.get("kilometers", 60000)),
        )
        X_list.append(x)
    X = pd.concat(X_list, ignore_index=True)
    return X

with st.spinner("Preparing feature matrix..."):
    X_eval = build_feature_matrix(sampled)

# True labels (encoded to model classes)
true_labels_str = sampled["price_band"].astype(str).values

try:
    y_true = label_encoder.transform(true_labels_str)
except Exception:
    # fallback: map unique labels to integers in sorted order
    uniq = sorted(list(pd.Series(true_labels_str).unique()))
    mapping = {v: i for i, v in enumerate(uniq)}
    y_true = np.array([mapping[v] for v in true_labels_str])

# Predictions
with st.spinner("Computing predictions and metrics..."):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)

# Classification report
n_classes = len(label_encoder.classes_) if hasattr(label_encoder, "classes_") else len(np.unique(np.concatenate([y_true, y_pred])))
class_names = list(label_encoder.classes_) if hasattr(label_encoder, "classes_") else [str(i) for i in range(n_classes)]

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

st.subheader("Classification Metrics")
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

st.divider()

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=class_names,
    y=class_names,
    colorscale="Viridis",
    hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
))
cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
st.plotly_chart(cm_fig, use_container_width=True)

st.divider()

# ROC Curves (one-vs-rest) for multiclass
st.subheader("ROC Curves (One-vs-Rest)")
try:
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig_roc = go.Figure()
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{class_names[i]} (AUC={roc_auc:.2f})"))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
    st.plotly_chart(fig_roc, use_container_width=True)

    overall_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    st.info(f"Macro-average AUC (OVR): {overall_auc:.3f}")
except Exception as e:
    st.warning(f"Could not compute ROC curves: {e}")

st.divider()

# Feature importance
st.subheader("Feature Importance")
if hasattr(model, "feature_importances_"):
    fi = np.asarray(model.feature_importances_)
    fn = feature_names
    fi_df = pd.DataFrame({"feature": fn, "importance": fi}).sort_values("importance", ascending=False).head(30)
    fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top Feature Importances")
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.write("Model does not expose feature_importances_.")

st.divider()

# Sample predictions
if show_sample:
    st.subheader("Sample Predictions")
    sample_out = sampled.copy()
    # Add predicted label and confidence
    try:
        pred_labels = label_encoder.inverse_transform(y_pred)
    except Exception:
        pred_labels = y_pred.astype(str)

    confidences = np.max(y_prob, axis=1)
    sample_out["predicted"] = pred_labels
    sample_out["confidence"] = np.round(confidences, 3)
    st.dataframe(sample_out[["title", "brand", "model", "price", "price_band", "predicted", "confidence"]].head(200), use_container_width=True)

st.success("Model performance metrics computed on the selected sample.")