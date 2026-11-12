
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Electricity Grid Analytics",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# 2. Load Model Bundle
# ==========================================

PROJECT_PATH = Path(r"D:\next Gen In Ai\Final Project")

MODEL_PATH = (
    PROJECT_PATH
    / "Models"
    / "electricity_grid_model.pkl"
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_bundle()

    model = bundle["model"]
    model_name = bundle["model_name"]

    history = bundle["history"].copy()
    metrics = pd.DataFrame(bundle["metrics"])

    features = bundle["features"]

    history["Datetime"] = pd.to_datetime(
        history["Datetime"]
    )

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# ==========================================
# 3. Sidebar Navigation
# ==========================================

st.sidebar.title("⚡ Electricity Grid")

st.sidebar.caption(
    "Electricity Consumption Analytics"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Analysis",
        "Forecasting",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Selected Model**")
st.sidebar.info(model_name)

# ==========================================
# 4. Overview Page
# ==========================================

if page == "Overview":

    st.title("Electricity Grid Analytics")

    st.markdown(
        "Monitor electricity consumption "
        "and explore forecasting insights."
    )

    # Calculate KPIs

    total_regions = history["Region"].nunique()

    total_readings = len(history)

    avg_consumption = history["Consumption"].mean()

    max_consumption = history["Consumption"].max()

    # KPI Cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Regions",
        f"{total_regions:,}"
    )

    col2.metric(
        "Total Readings",
        f"{total_readings:,}"
    )

    col3.metric(
        "Average Consumption",
        f"{avg_consumption:,.2f}"
    )

    col4.metric(
        "Maximum Consumption",
        f"{max_consumption:,.2f}"
    )

    st.divider()

    # Consumption Chart

    st.subheader("Electricity Consumption Trends")

    selected_region = st.selectbox(
        "Select Region",
        sorted(history["Region"].unique())
    )

    region_data = history[
        history["Region"] == selected_region
    ].sort_values("Datetime")

    fig = px.line(
        region_data.tail(500),
        x="Datetime",
        y="Consumption",
        title=f"Consumption - {selected_region}",
        template="plotly_white"
    )

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Project Information")

    st.write("Best Model:", model_name)

    st.write("Number of Features:", len(features))

# ==========================================
# 5. Data Analysis Page
# ==========================================

elif page == "Data Analysis":

    st.title("Data Analysis")

    st.markdown(
        "Explore electricity consumption "
        "across different regions and time periods."
    )

    regions = sorted(history["Region"].unique())

    selected_region = st.selectbox(
        "Choose Region",
        regions
    )

    region_data = history[
        history["Region"] == selected_region
    ].copy()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Average Consumption",
            f"{region_data['Consumption'].mean():,.2f}"
        )

    with col2:

        st.metric(
            "Total Records",
            f"{len(region_data):,}"
        )

    # Daily consumption

    st.subheader("Daily Consumption")

    daily_data = (
        region_data
        .set_index("Datetime")["Consumption"]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig_daily = px.line(
        daily_data,
        x="Datetime",
        y="Consumption",
        title="Average Daily Consumption",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )

    # Consumption distribution

    st.subheader("Consumption Distribution")

    fig_hist = px.histogram(
        region_data,
        x="Consumption",
        nbins=50,
        title="Consumption Distribution",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    # Raw Data

    with st.expander("View Raw Data"):

        st.dataframe(
            region_data.head(100),
            use_container_width=True
        )

# ==========================================
# 6. Forecasting Page
# ==========================================

elif page == "Forecasting":

    st.title("Electricity Consumption Forecasting")

    st.info(
        "Forecasting functionality will be added "
        "after connecting the model with the "
        "historical lag and rolling features."
    )

    st.write("Selected Model:", model_name)

    st.write("Available Regions:")

    st.write(sorted(history["Region"].unique()))

# ==========================================
# 7. Model Performance Page
# ==========================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown(
        "Compare the performance of trained "
        "machine learning models."
    )

    st.subheader("Evaluation Metrics")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    if "RMSE" in metrics.columns and "Model" in metrics.columns:

        fig = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            color="Model",
            title="Model Comparison - RMSE",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "R2" in metrics.columns and "Model" in metrics.columns:

        fig = px.bar(
            metrics,
            x="Model",
            y="R2",
            color="Model",
            title="Model Comparison - R²",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.success(
        f"Best Model: {model_name}"
    )
