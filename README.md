# UsedCarDSS — Used Car Price Decision Support System

A Streamlit app for exploring used-car data and predicting vehicle prices using machine learning models. This repository contains data, model artifacts, visualization utilities, and an explainability pipeline (SHAP) to help users understand model predictions.

## Features

- Interactive Streamlit multi-page app for:
  - Price prediction
  - Model performance visualization
  - Explainable AI (SHAP) insights
  - Market insights and visualizations
- End-to-end pipeline: data loading, preprocessing, model inference
- Modular utilities for charts, preprocessing, and explainability

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app locally:

```bash
streamlit run app.py
```

3. Open the shown local URL (usually `http://localhost:8501`).

## Project Structure

- `requirements.txt` — Python dependencies
- `assets/` — styling and static assets (e.g., `style.css`)
- `data/used_car.csv` — dataset (sample or full)
- `images/` — images used in the app
- `models/` — serialized model artifacts (pickle, joblib, or similar)
- `outputs/` — generated outputs, reports, or exported figures
- `pages/` — Streamlit multipage scripts:
  - `01_Home.py`
  - `02_Price_prediction.py`
  - `03_Model_Performance.py`
  - `04_Explainable_AI.py`
  - `05_Market_Insights.py`
  - `06_About.py`
- `utils/` — helper modules:
  - `loader.py` — data loading utilities
  - `preprocessing.py` — feature processing and transforms
  - `predictor.py` — model inference wrappers
  - `explainer.py` / `shap_utils.py` — SHAP explainability helpers
  - `charts.py` — plotting utilities
  - `constants.py` — configuration and constants

## Data

- The dataset is expected at `data/used_car.csv`.
- Ensure the file has the columns used by `preprocessing.py` and model training/inference.
- If you need to retrain models, create a training script or notebook that reads from `data/` and stores artifacts in `models/`.

## Models & Training

- Trained models (if provided) live in the `models/` folder. Typical formats: `.pkl`, `.joblib`, or exported scikit-learn pipelines.
- `predictor.py` wraps the model and preprocessing pipeline to produce price predictions from raw inputs.
- To retrain, follow your existing training notebook or implement a new training script; save the fitted pipeline to `models/`.

## Explainability (SHAP)

- The `explainer.py` and `shap_utils.py` modules compute and visualize SHAP values for individual predictions and global feature importance.
- Use the `04_Explainable_AI.py` page to interact with SHAP explanations in the app.

## Deployment

- The app can be deployed to Streamlit Cloud, Heroku, or other container platforms.
- For Streamlit Cloud, push the repository to GitHub and connect a new app using the `app.py` entrypoint.

## Development Tips

- Keep `models/` out of version control if artifacts are large; use `.gitignore` to exclude them and provide small sample artifacts for demos.
- Use relative paths and the `utils/loader.py` helpers to load data and models consistently across pages.
- Add unit tests for `preprocessing.py` and `predictor.py` to avoid regressions.

## Contributing

- Fork the repo, create a feature branch, and open a PR with a clear description.
- Please run linting and tests before submitting changes.

## License

Specify your license here (e.g., MIT). If you don't want a license, state that the repository is proprietary.

---

Generated README for quick onboarding and usage. Update sections (Data, Models, License) with project-specific details as needed.

