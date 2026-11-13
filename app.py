import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Power Grid Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_REPO_ID = "AhmedN3yazi/electricity-grid-model"
HF_MODEL_FILENAME = "electricity_grid_model.pkl"

# Professional dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0b1220;
    }
    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #263449;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142238, #101a2b);
        border: 1px solid #263b55;
        padding: 17px 18px;
        border-radius: 13px;
    }
    [data-testid="stMetricLabel"] {
        color: #a9bad0;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #f4f7fb;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f4f7fb;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .sub-title {
        color: #91a4bd;
        margin-top: 4px;
        margin-bottom: 22px;
    }
    .section-title {
        color: #e8eef7;
        font-size: 1.12rem;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .info-card {
        background: #142238;
        border: 1px solid #263b55;
        padding: 16px;
        border-radius: 13px;
    }
    div[data-testid="stPlotlyChart"] {
        background: #101a2b;
        border: 1px solid #263449;
        border-radius: 13px;
        padding: 8px;
    }
    .small-muted {
        color: #91a4bd;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
CHART_CONFIG = {"displayModeBar": True, "displaylogo": False, "responsive": True}

HF_REPO_ID = "AhmedN3yazi/electricity-grid-model"
HF_MODEL_FILENAME = "dashboard_data.pkl"

# =========================================================
# 2. LOAD THE SAVED MODEL BUNDLE
# =========================================================
@st.cache_resource
def load_bundle():
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

    if not token:
        raise RuntimeError(
            "Hugging Face token is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    data_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model",
        token=token,
    )

    return joblib.load(data_path)


try:
    bundle = load_bundle()
    model_name = bundle["model_name"]
    history = bundle["history"].copy()
    metrics = pd.DataFrame(bundle["metrics"])

except Exception as error:
    st.error(f"Could not load dashboard data: {error}")
    st.stop()

required_columns = {"Datetime", "Region", "Consumption"}
missing_columns = required_columns - set(history.columns)
if missing_columns:
    st.error(f"Missing columns in saved history: {sorted(missing_columns)}")
    st.stop()

history["Datetime"] = pd.to_datetime(history["Datetime"], errors="coerce")
history["Consumption"] = pd.to_numeric(history["Consumption"], errors="coerce")
history = history.dropna(subset=["Datetime", "Region", "Consumption"]).copy()

# Derived calendar fields for dashboard analysis
history["Hour"] = history["Datetime"].dt.hour
history["Day"] = history["Datetime"].dt.day
history["DayOfWeek"] = history["Datetime"].dt.dayofweek
history["DayName"] = history["Datetime"].dt.day_name()
history["Month"] = history["Datetime"].dt.month
history["MonthName"] = history["Datetime"].dt.month_name()
history["Year"] = history["Datetime"].dt.year
history["Date"] = history["Datetime"].dt.date
history["Quarter"] = history["Datetime"].dt.quarter

if "Occurrence" not in history.columns:
    history["Occurrence"] = 0

regions = sorted(history["Region"].astype(str).unique().tolist())
min_date = history["Datetime"].min().date()
max_date = history["Datetime"].max().date()

# Consistent Plotly layout
def polish(fig, height=390):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#dce6f2"),
        margin=dict(l=18, r=18, t=58, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#17263b", font_size=12)
    )
    fig.update_xaxes(gridcolor="#263449", zerolinecolor="#263449")
    fig.update_yaxes(gridcolor="#263449", zerolinecolor="#263449")
    return fig

def page_header(title, subtitle):
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)

def format_number(value):
    return f"{value:,.2f}" if pd.notna(value) else "N/A"


# =========================================================
# 3. SIDEBAR + GLOBAL FILTERS
# =========================================================
st.sidebar.markdown("# ⚡ POWER GRID")
st.sidebar.caption("ELECTRICITY INTELLIGENCE PLATFORM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "WORKSPACE",
    ["Executive Overview", "Regional Analysis", "Time Intelligence",
     "Consumption Explorer", "Model Performance", "Forecasting"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### GLOBAL FILTERS")

selected_regions = st.sidebar.multiselect(
    "Select regions",
    options=regions,
    default=regions
)

if not selected_regions:
    st.warning("Select at least one region from the sidebar.")
    st.stop()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = history[
    history["Region"].astype(str).isin(selected_regions)
    & history["Datetime"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Best model: {model_name}")
st.sidebar.caption(f"Regions selected: {len(selected_regions)}")


# =========================================================
# 4. EXECUTIVE OVERVIEW
# =========================================================
if page == "Executive Overview":
    page_header(
        "Electricity Grid | Executive Overview",
        "A high-level view of electricity demand, regional behavior and historical trends."
    )

    total_records = len(filtered)
    avg_consumption = filtered["Consumption"].mean()
    peak_consumption = filtered["Consumption"].max()
    min_consumption = filtered["Consumption"].min()
    total_regions = filtered["Region"].nunique()
    median_consumption = filtered["Consumption"].median()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Readings", f"{total_records:,}")
    k2.metric("Regions", f"{total_regions:,}")
    k3.metric("Average Demand", format_number(avg_consumption))
    k4.metric("Peak Demand", format_number(peak_consumption))
    k5.metric("Median Demand", format_number(median_consumption))

    st.markdown('<div class="section-title">Demand Overview</div>', unsafe_allow_html=True)

    daily = (
        filtered.set_index("Datetime")
        .groupby("Region")["Consumption"]
        .resample("D").mean()
        .reset_index()
    )

    c1, c2 = st.columns([1.65, 1])
    with c1:
        fig = px.line(
            daily, x="Datetime", y="Consumption", color="Region",
            title="Daily Average Electricity Consumption",
            labels={"Datetime": "Date", "Consumption": "Average consumption"}
        )
        st.plotly_chart(polish(fig, 430), use_container_width=True, config=CHART_CONFIG)

    with c2:
        region_avg = (
            filtered.groupby("Region", as_index=False)["Consumption"]
            .mean().sort_values("Consumption", ascending=True)
        )
        fig = px.bar(
            region_avg, x="Consumption", y="Region", orientation="h",
            color="Consumption", color_continuous_scale="Teal",
            title="Average Demand by Region"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig, 430), use_container_width=True, config=CHART_CONFIG)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(
            filtered, x="Consumption", nbins=55,
            title="Electricity Consumption Distribution",
            marginal="box", color_discrete_sequence=["#28b8a6"]
        )
        st.plotly_chart(polish(fig, 390), use_container_width=True, config=CHART_CONFIG)

    with c4:
        box_data = filtered.sample(min(len(filtered), 30000), random_state=42)
        fig = px.box(
            box_data, x="Region", y="Consumption", color="Region",
            title="Demand Spread and Outliers by Region",
            points=False
        )
        st.plotly_chart(polish(fig, 390), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Key Regional Statistics</div>', unsafe_allow_html=True)
    summary = (
        filtered.groupby("Region")
        .agg(
            Readings=("Consumption", "count"),
            Mean=("Consumption", "mean"),
            Median=("Consumption", "median"),
            Std=("Consumption", "std"),
            Minimum=("Consumption", "min"),
            Maximum=("Consumption", "max")
        )
        .round(2).sort_values("Mean", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)


# =========================================================
# 5. REGIONAL ANALYSIS
# =========================================================
elif page == "Regional Analysis":
    page_header(
        "Regional Intelligence",
        "Compare grid regions and identify differences in demand level, volatility and load shape."
    )

    regional = (
        filtered.groupby("Region")
        .agg(
            Mean=("Consumption", "mean"),
            Median=("Consumption", "median"),
            Std=("Consumption", "std"),
            Min=("Consumption", "min"),
            Max=("Consumption", "max"),
            Records=("Consumption", "count")
        ).reset_index()
    )
    regional["Coefficient of Variation"] = (
        regional["Std"] / regional["Mean"].replace(0, np.nan)
    ) * 100

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            regional.sort_values("Mean"), x="Mean", y="Region",
            orientation="h", color="Mean", color_continuous_scale="Viridis",
            title="Average Consumption by Region"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)
    with b:
        fig = px.bar(
            regional.sort_values("Coefficient of Variation"),
            x="Coefficient of Variation", y="Region",
            orientation="h", color="Coefficient of Variation",
            color_continuous_scale="Plasma",
            title="Relative Demand Volatility (CV %)"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Regional Demand Profile</div>', unsafe_allow_html=True)
    hourly_region = (
        filtered.groupby(["Region", "Hour"], as_index=False)["Consumption"].mean()
    )
    fig = px.line(
        hourly_region, x="Hour", y="Consumption", color="Region",
        markers=True, title="Average 24-Hour Load Shape by Region"
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(polish(fig, 440), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Regional Performance Matrix</div>', unsafe_allow_html=True)
    metric_choice = st.selectbox(
        "Heatmap metric",
        ["Mean", "Median", "Std", "Coefficient of Variation", "Records"]
    )
    fig = px.imshow(
        regional.set_index("Region")[[metric_choice]].T,
        aspect="auto", color_continuous_scale="Teal",
        title=f"Regional Comparison — {metric_choice}",
        labels=dict(x="Region", y="Metric", color=metric_choice)
    )
    st.plotly_chart(polish(fig, 230), use_container_width=True, config=CHART_CONFIG)

    st.dataframe(
        regional.round(2).sort_values("Mean", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 6. TIME INTELLIGENCE
# =========================================================
elif page == "Time Intelligence":
    page_header(
        "Time Intelligence",
        "Explore hourly, weekly, monthly and seasonal electricity demand patterns."
    )

    hourly = filtered.groupby("Hour", as_index=False)["Consumption"].mean()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    weekly = filtered.groupby("DayName", as_index=False)["Consumption"].mean()
    weekly["DayName"] = pd.Categorical(weekly["DayName"], weekday_order, ordered=True)
    weekly = weekly.sort_values("DayName")
    monthly = filtered.groupby("Month", as_index=False)["Consumption"].mean()
    monthly["MonthName"] = monthly["Month"].map({
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
    })

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            hourly, x="Hour", y="Consumption", markers=True,
            title="Average Hourly Demand Profile",
            color_discrete_sequence=["#2dd4bf"]
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)
    with c2:
        fig = px.bar(
            weekly, x="DayName", y="Consumption",
            title="Average Demand by Day of Week",
            color="Consumption", color_continuous_scale="Teal"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(
            monthly, x="MonthName", y="Consumption", markers=True,
            title="Monthly Seasonality",
            category_orders={"MonthName": ["Jan","Feb","Mar","Apr","May","Jun",
                                           "Jul","Aug","Sep","Oct","Nov","Dec"]}
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)
    with c4:
        heat = filtered.pivot_table(
            index="DayName", columns="Hour", values="Consumption",
            aggfunc="mean", observed=False
        ).reindex(weekday_order)
        fig = px.imshow(
            heat, aspect="auto", color_continuous_scale="Turbo",
            title="Hourly Demand Heatmap by Weekday",
            labels=dict(x="Hour of day", y="Day of week", color="Demand")
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Yearly Demand Evolution</div>', unsafe_allow_html=True)
    yearly = filtered.groupby(["Year", "Region"], as_index=False)["Consumption"].mean()
    fig = px.line(
        yearly, x="Year", y="Consumption", color="Region", markers=True,
        title="Annual Average Consumption by Region"
    )
    st.plotly_chart(polish(fig, 420), use_container_width=True, config=CHART_CONFIG)


# =========================================================
# 7. CONSUMPTION EXPLORER
# =========================================================
elif page == "Consumption Explorer":
    page_header(
        "Consumption Explorer",
        "Investigate the distribution, extremes and detailed records behind electricity demand."
    )

    region_options = ["All selected regions"] + sorted(filtered["Region"].unique().tolist())
    chosen_region = st.selectbox("Focus region", region_options)
    explore = filtered.copy()
    if chosen_region != "All selected regions":
        explore = explore[explore["Region"] == chosen_region]

    c1, c2, c3 = st.columns(3)
    c1.metric("Minimum", format_number(explore["Consumption"].min()))
    c2.metric("Maximum", format_number(explore["Consumption"].max()))
    c3.metric("Standard Deviation", format_number(explore["Consumption"].std()))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            explore, x="Consumption", nbins=70, marginal="violin",
            title="Consumption Frequency Distribution",
            color_discrete_sequence=["#60a5fa"]
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)
    with c2:
        fig = px.violin(
            explore.sample(min(len(explore), 25000), random_state=42),
            x="Region", y="Consumption", color="Region",
            box=True, points=False, title="Consumption Density by Region"
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            explore.sample(min(len(explore), 12000), random_state=42),
            x="Hour", y="Consumption", color="Region",
            opacity=0.45, title="Consumption vs Hour of Day"
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)
    with c4:
        fig = px.scatter(
            explore.sample(min(len(explore), 12000), random_state=42),
            x="DayOfWeek", y="Consumption", color="Region",
            opacity=0.45, title="Consumption vs Day of Week"
        )
        fig.update_xaxes(
            tickmode="array", tickvals=list(range(7)),
            ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        )
        st.plotly_chart(polish(fig), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Highest and Lowest Demand Records</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    display_cols = ["Datetime", "Region", "Consumption"]
    with left:
        st.markdown("**Top 20 peak readings**")
        st.dataframe(
            explore.nlargest(20, "Consumption")[display_cols],
            use_container_width=True, hide_index=True
        )
    with right:
        st.markdown("**Bottom 20 readings**")
        st.dataframe(
            explore.nsmallest(20, "Consumption")[display_cols],
            use_container_width=True, hide_index=True
        )

    with st.expander("Explore filtered records"):
        st.dataframe(
            explore.sort_values("Datetime", ascending=False),
            use_container_width=True,
            hide_index=True
        )
        csv = explore.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data (CSV)",
            data=csv,
            file_name="electricity_filtered_data.csv",
            mime="text/csv"
        )


# =========================================================
# 8. MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    page_header(
        "Machine Learning Model Performance",
        "Compare the trained regressors using the evaluation metrics generated in the notebook."
    )

    if metrics.empty:
        st.warning("No model metrics were found in the saved bundle.")
        st.stop()

    st.markdown(f"**Selected best model:** `{model_name}`")

    metric_cols = [c for c in ["MAE", "RMSE", "R2"] if c in metrics.columns]
    if "Model" not in metrics.columns or not metric_cols:
        st.error("The saved metrics table must contain Model and evaluation metric columns.")
        st.dataframe(metrics, use_container_width=True)
        st.stop()

    best_row = metrics.loc[metrics["RMSE"].idxmin()] if "RMSE" in metrics.columns else None
    if best_row is not None:
        a, b, c = st.columns(3)
        a.metric("Best Model", str(best_row["Model"]))
        b.metric("Lowest RMSE", format_number(best_row["RMSE"]))
        c.metric("MAE", format_number(best_row["MAE"]) if "MAE" in metrics.columns else "N/A")

    st.markdown('<div class="section-title">Model Metrics Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(len(metric_cols))
    for col, metric in zip(cols, metric_cols):
        with col:
            ascending = metric != "R2"
            ordered = metrics.sort_values(metric, ascending=ascending)
            fig = px.bar(
                ordered, x="Model", y=metric, color="Model",
                title=f"{metric} Comparison",
                text_auto=".3f"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish(fig, 360), use_container_width=True, config=CHART_CONFIG)

    if {"MAE", "RMSE", "R2"}.issubset(metrics.columns):
        st.markdown('<div class="section-title">Metric Relationship</div>', unsafe_allow_html=True)
        fig = px.scatter(
            metrics, x="RMSE", y="MAE", size="R2",
            color="Model", text="Model",
            title="MAE vs RMSE (bubble size represents R²)"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(polish(fig, 400), use_container_width=True, config=CHART_CONFIG)

    st.markdown('<div class="section-title">Full Evaluation Table</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics.sort_values("RMSE") if "RMSE" in metrics.columns else metrics,
        use_container_width=True, hide_index=True
    )

    st.caption(
        "Metrics shown here are the saved test-set evaluation results from the training notebook."
    )


# =========================================================
# 9. FORECASTING WORKSPACE
# =========================================================
elif page == "Forecasting":
    page_header(
        "Electricity Demand Forecasting",
        "Forecasting workspace — connect future feature generation to the saved trained pipeline."
    )

    st.info(
        "The model uses historical lag and rolling features. "
        "A valid multi-step forecast must generate these features recursively for each region."
    )

    c1, c2 = st.columns(2)
    with c1:
        forecast_region = st.selectbox(
            "Choose a region",
            sorted(filtered["Region"].unique())
        )
    with c2:
        horizon = st.selectbox(
            "Forecast horizon",
            [24, 48, 72, 168],
            format_func=lambda x: f"{x} hours"
        )

    region_history = (
        history[history["Region"] == forecast_region]
        .sort_values(["Datetime", "Occurrence"])
    )

    st.metric("Selected Model", model_name)
    st.metric("Available historical readings", f"{len(region_history):,}")

    st.markdown("#### Latest historical observations")
    st.dataframe(
        region_history[["Datetime", "Consumption"]].tail(15),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Forecast generation is intentionally not enabled yet. "
        "Before producing forecasts, we must confirm the exact timestamp frequency, "
        "Occurrence handling, and recursive lag/rolling feature logic used by the notebook. "
        "This avoids displaying fabricated or incorrectly engineered predictions."
    )

# =========================================================
# 10. FOOTER
# =========================================================
st.sidebar.markdown("---")
st.sidebar.caption("POWER GRID ANALYTICS")
st.sidebar.caption("Machine Learning • Time Series • Energy Intelligence")
