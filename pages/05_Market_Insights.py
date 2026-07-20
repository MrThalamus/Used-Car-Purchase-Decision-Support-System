import streamlit as st
import pandas as pd
import plotly.express as px

from utils import loader, charts

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(page_title="Market Insights", layout="wide")

st.title("Market Insights")
st.markdown("Explore market-level statistics, distributions, and trends for the used car dataset.")
st.divider()

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
@st.cache_data
def get_data():
    df = loader.load_dataset()
    # Ensure expected columns exist; if year missing try to infer
    if "year" not in df.columns and "car_age" in df.columns:
        # approximate manufacturing year from car_age (assuming current year if available)
        try:
            import datetime

            current_year = datetime.datetime.now().year
            df["year"] = current_year - df["car_age"]
        except Exception:
            df["year"] = None
    return df

df = get_data()

# ----------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------
st.sidebar.header("Filters")

brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Brand", brands)

transmissions = ["All"] + sorted(df["transmission"].dropna().unique().tolist())
selected_transmission = st.sidebar.selectbox("Transmission", transmissions)

min_year = int(df["year"].min()) if df["year"].notna().any() else 2000
max_year = int(df["year"].max()) if df["year"].notna().any() else 2026
selected_year_range = st.sidebar.slider("Manufacturing Year Range", min_year, max_year, (min_year, max_year))

# Apply filters
filtered = df.copy()
if selected_brand != "All":
    filtered = filtered[filtered["brand"] == selected_brand]

if selected_transmission != "All":
    filtered = filtered[filtered["transmission"] == selected_transmission]

if "year" in filtered.columns and filtered["year"].notna().any():
    filtered = filtered[(filtered["year"] >= selected_year_range[0]) & (filtered["year"] <= selected_year_range[1])]

# ----------------------------------------------------------
# Overview metrics
# ----------------------------------------------------------
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Listings", f"{len(filtered):,}")
with col2:
    st.metric("Average Price", f"BDT {filtered['price'].mean():,.0f}")
with col3:
    st.metric("Median Price", f"BDT {filtered['price'].median():,.0f}")
with col4:
    if "kilometers_run" in filtered.columns:
        st.metric("Average Mileage", f"{filtered['kilometers_run'].mean():,.0f} km")

st.divider()

# ----------------------------------------------------------
# Top row charts
# ----------------------------------------------------------
st.subheader("Distributions")
row1_col1, row1_col2, row1_col3 = st.columns((1.2, 1, 1))

with row1_col1:
    st.plotly_chart(charts.price_distribution(filtered), use_container_width=True)
with row1_col2:
    st.plotly_chart(charts.price_band_distribution(filtered), use_container_width=True)
with row1_col3:
    st.plotly_chart(charts.brand_distribution(filtered), use_container_width=True)

st.divider()

# ----------------------------------------------------------
# Second row charts
# ----------------------------------------------------------
st.subheader("Market Breakdown")
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.plotly_chart(charts.fuel_distribution(filtered), use_container_width=True)
with row2_col2:
    st.plotly_chart(charts.transmission_distribution(filtered), use_container_width=True)
with row2_col3:
    st.plotly_chart(charts.average_price_by_brand(filtered), use_container_width=True)

st.divider()

# ----------------------------------------------------------
# Relationship charts
# ----------------------------------------------------------
st.subheader("Relationships")
rel_col1, rel_col2 = st.columns(2)

with rel_col1:
    st.plotly_chart(charts.age_vs_price(filtered), use_container_width=True)
with rel_col2:
    st.plotly_chart(charts.mileage_vs_price(filtered), use_container_width=True)

st.divider()

# ----------------------------------------------------------
# Time series: average price by year
# ----------------------------------------------------------
if "year" in filtered.columns and filtered["year"].notna().any():
    st.subheader("Trends: Average Price by Year")
    ts = (
        filtered.groupby("year")["price"].mean().reset_index().sort_values("year")
    )
    fig_ts = px.line(ts, x="year", y="price", markers=True, title="Average Price by Manufacturing Year")
    st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ----------------------------------------------------------
# Data preview and download
# ----------------------------------------------------------
st.subheader("Data Preview")
st.dataframe(filtered.head(200), use_container_width=True)

@st.cache_data

def to_csv(df_):
    return df_.to_csv(index=False).encode("utf-8")

st.download_button("Download Filtered Data", data=to_csv(filtered), file_name="market_insights_filtered.csv")