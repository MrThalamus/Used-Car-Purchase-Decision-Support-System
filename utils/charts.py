import plotly.express as px
import plotly.graph_objects as go


def brand_distribution(df):
    data = (
        df["brand"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    data.columns = ["Brand", "Count"]

    fig = px.bar(
        data,
        x="Brand",
        y="Count",
        color="Count",
        text="Count",
        title="Top 15 Brands"
    )

    fig.update_layout(coloraxis_showscale=False)

    return fig


def fuel_distribution(df):

    data = (
        df["fuel_type"]
        .value_counts()
        .reset_index()
    )

    data.columns = ["Fuel", "Count"]

    fig = px.pie(
        data,
        names="Fuel",
        values="Count",
        hole=0.45,
        title="Fuel Type Distribution"
    )

    return fig


def transmission_distribution(df):

    data = (
        df["transmission"]
        .value_counts()
        .reset_index()
    )

    data.columns = ["Transmission", "Count"]

    fig = px.pie(
        data,
        names="Transmission",
        values="Count",
        hole=0.45,
        title="Transmission Distribution"
    )

    return fig


def price_distribution(df):

    fig = px.histogram(
        df,
        x="price",
        nbins=40,
        title="Price Distribution"
    )

    return fig


def price_band_distribution(df):

    data = (
        df["price_band"]
        .value_counts()
        .reset_index()
    )

    data.columns = ["Band", "Count"]

    fig = px.bar(
        data,
        x="Band",
        y="Count",
        color="Band",
        text="Count",
        title="Price Category Distribution"
    )

    return fig


def average_price_by_brand(df):

    data = (
        df.groupby("brand")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        data,
        x="brand",
        y="price",
        color="price",
        text_auto=".2s",
        title="Average Price by Brand"
    )

    fig.update_layout(coloraxis_showscale=False)

    return fig


def age_vs_price(df):

    fig = px.scatter(
        df,
        x="car_age",
        y="price",
        color="price_band",
        title="Car Age vs Price"
    )

    return fig


def mileage_vs_price(df):

    fig = px.scatter(
        df,
        x="kilometers_run",
        y="price",
        color="price_band",
        title="Mileage vs Price"
    )

    return fig


def engine_vs_price(df):

    fig = px.scatter(
        df,
        x="energy_capacity",
        y="price",
        color="price_band",
        title="Engine Capacity vs Price"
    )

    return fig