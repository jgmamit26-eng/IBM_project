# =============================================================================
# TrafficPulse AI — Traffic, Congestion & Accident Analytics
# Author  : Amit Dutta  |  Report Author: Rajshekhar Jana
# Version : 1.0
# Run     : streamlit run "Amit Dutta_TrafficPulseAI.py"
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

# ── scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TrafficPulse AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }

    /* KPI card style */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin-bottom: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #8892b0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-icon { font-size: 1.5rem; margin-bottom: 4px; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ccd6f6;
        border-left: 4px solid #4fc3f7;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Insight box */
    .insight-box {
        background: #1a2332;
        border: 1px solid #1e6091;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #a8dadc;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Alert boxes */
    .alert-high {
        background: #2d1515;
        border-left: 4px solid #e74c3c;
        border-radius: 6px;
        padding: 12px 16px;
        color: #f1948a;
        margin: 6px 0;
    }
    .alert-medium {
        background: #2d2515;
        border-left: 4px solid #f39c12;
        border-radius: 6px;
        padding: 12px 16px;
        color: #f8c471;
        margin: 6px 0;
    }
    .alert-low {
        background: #152d1a;
        border-left: 4px solid #27ae60;
        border-radius: 6px;
        padding: 12px 16px;
        color: #82e0aa;
        margin: 6px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px 8px 0 0;
        color: #8892b0;
        padding: 8px 18px;
        border: 1px solid #2e3450;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4fc3f7 !important;
        color: #0f1117 !important;
        font-weight: 700;
    }

    /* Plotly chart container */
    .plotly-chart { border-radius: 10px; overflow: hidden; }

    /* Sidebar branding */
    .sidebar-brand {
        text-align: center;
        padding: 16px 0 24px 0;
        border-bottom: 1px solid #2e3450;
        margin-bottom: 16px;
    }
    .sidebar-brand h2 {
        color: #4fc3f7;
        font-size: 1.4rem;
        margin: 0;
    }
    .sidebar-brand p {
        color: #8892b0;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e6091, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #4fc3f7; font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { color: #8892b0; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1d27; }
    ::-webkit-scrollbar-thumb { background: #2e3450; border-radius: 3px; }

    h1, h2, h3 { color: #ccd6f6 !important; }
    p, li, label { color: #a8b2d8; }
    .stMarkdown p { color: #a8b2d8; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY THEME
# =============================================================================
PLOTLY_THEME = "plotly_dark"
CHART_BG = "#1a1d27"
GRID_COLOR = "#2e3450"

def styled_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_THEME,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color="#ccd6f6", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GRID_COLOR, borderwidth=1),
    )
    return fig

# =============================================================================
# SYNTHETIC DATA GENERATOR  (fallback when no CSV is uploaded)
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=5000):
    """Generate realistic synthetic traffic data matching the project schema."""
    rng = np.random.default_rng(42)
    locations = [
        "Salt Lake", "Jadavpur", "Gariahat", "New Town", "Howrah Bridge",
        "Sealdah", "Esplanade", "EM Bypass", "Ballygunge", "Dum Dum",
        "Park Street", "Rajarhat"
    ]
    loc_coords = {
        "Salt Lake": (22.5726, 88.4195), "Jadavpur": (22.4990, 88.3720),
        "Gariahat": (22.5212, 88.3693), "New Town": (22.5958, 88.4785),
        "Howrah Bridge": (22.5851, 88.3468), "Sealdah": (22.5647, 88.3700),
        "Esplanade": (22.5627, 88.3509), "EM Bypass": (22.5040, 88.3900),
        "Ballygunge": (22.5260, 88.3638), "Dum Dum": (22.6190, 88.4215),
        "Park Street": (22.5512, 88.3511), "Rajarhat": (22.6085, 88.4579),
    }
    road_types = ["Arterial", "Highway", "Collector", "Local"]
    road_conditions = ["Good", "Fair", "Poor", "Wet"]
    weathers = ["Clear", "Rain", "Fog", "Heavy Rain", "Cloudy"]
    congestion_levels = ["Low", "Medium", "High", "Severe"]
    signal_statuses = ["Normal", "Disrupted", "Off"]
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    start = datetime(2024, 1, 1)
    dates = [start + timedelta(hours=int(i)) for i in rng.integers(0, 12000, n)]
    locs  = rng.choice(locations, n)
    hours = np.array([d.hour for d in dates])

    # Traffic volume — peaks at 8-10 am and 5-7 pm
    base_vol = 100 + 150 * (
        np.exp(-((hours - 8.5)**2) / 4) + np.exp(-((hours - 17.5)**2) / 4)
    )
    vehicle_count = (base_vol + rng.normal(0, 30, n)).clip(10, 500).astype(int)

    weather_arr = rng.choice(weathers, n, p=[0.55, 0.16, 0.04, 0.06, 0.19])
    road_cond = rng.choice(road_conditions, n, p=[0.49, 0.28, 0.09, 0.14])
    road_type = rng.choice(road_types, n, p=[0.45, 0.15, 0.26, 0.14])

    temp = rng.uniform(18, 40, n)
    visibility = np.where(
        weather_arr == "Fog", rng.uniform(0.5, 3, n),
        np.where(weather_arr == "Heavy Rain", rng.uniform(2, 6, n),
                 rng.uniform(5, 12, n))
    )

    congestion_prob = (vehicle_count / 500 * 0.6
                       + (weather_arr == "Heavy Rain") * 0.2
                       + (weather_arr == "Fog") * 0.15
                       + (road_cond == "Poor") * 0.1)
    congestion_idx = np.zeros(n, int)
    for i, p in enumerate(congestion_prob):
        congestion_idx[i] = rng.choice([0,1,2,3], p=np.array(
            [max(0,(1-p)*0.4), max(0,(1-p)*0.4+p*0.3),
             max(0,p*0.3+0.1), max(0,p*0.1+0.05)]
        ) / np.array([max(0,(1-p)*0.4), max(0,(1-p)*0.4+p*0.3),
                       max(0,p*0.3+0.1), max(0,p*0.1+0.05)]).sum())
    congestion = np.array(congestion_levels)[congestion_idx]

    avg_speed = (70 - 20*(congestion_idx/3) + rng.normal(0,5,n)).clip(5,100)
    accident_prob = (0.015 + 0.04*(weather_arr=="Heavy Rain")
                     + 0.03*(road_cond=="Poor") + 0.02*(congestion_idx>=2)
                     + 0.01*(visibility<2))
    accident = (rng.random(n) < accident_prob).astype(int)
    severity_map = {0: "None", 1: "Minor", 2: "Moderate", 3: "Severe"}
    acc_severity = np.where(accident==0, "None",
                   np.array(["Minor","Moderate","Severe"])[
                       rng.choice(3, n, p=[0.6,0.3,0.1])])

    traffic_density = (vehicle_count / 10 + rng.normal(0, 2, n)).clip(0)
    travel_time     = (10 - avg_speed/15 + rng.normal(0,1,n)).clip(1, 30)
    rainfall        = np.where(np.isin(weather_arr, ["Rain","Heavy Rain"]),
                               rng.uniform(0.5, 20, n), 0.0)
    parking_occ     = rng.uniform(20, 95, n)
    pop_density     = rng.integers(5000, 20000, n)
    signal_status   = rng.choice(signal_statuses, n, p=[0.85, 0.10, 0.05])
    holiday         = (rng.random(n) < 0.06).astype(int)
    special_event   = (rng.random(n) < 0.04).astype(int)
    aqi             = rng.uniform(30, 300, n)
    ert             = np.where(accident==1, rng.uniform(5, 40, n), np.nan)

    rows = []
    for i in range(n):
        d = dates[i]
        loc = locs[i]
        lat, lon = loc_coords[loc]
        lat += rng.uniform(-0.005, 0.005)
        lon += rng.uniform(-0.005, 0.005)
        rows.append({
            "Record_ID": i+1,
            "Date": d.date(),
            "Time": d.strftime("%H:%M:%S"),
            "Day_of_Week": days[d.weekday()],
            "Month": d.month,
            "Year": d.year,
            "Hour": d.hour,
            "Location": loc,
            "Latitude": round(lat, 6),
            "Longitude": round(lon, 6),
            "Road_Type": road_type[i],
            "Road_Condition": road_cond[i],
            "Weather": weather_arr[i],
            "Temperature_C": round(temp[i], 1),
            "Rainfall_mm": round(rainfall[i], 2),
            "Visibility_km": round(visibility[i], 2),
            "Traffic_Volume": vehicle_count[i],
            "Average_Speed_kmph": round(avg_speed[i], 1),
            "Vehicle_Count": vehicle_count[i],
            "Traffic_Density": round(traffic_density[i], 2),
            "Travel_Time_min": round(travel_time[i], 2),
            "Congestion_Level": congestion[i],
            "Accident_Occurred": accident[i],
            "Accident_Severity": acc_severity[i],
            "Emergency_Response_Time": round(ert[i], 1) if not np.isnan(ert[i]) else None,
            "Traffic_Signal_Status": signal_status[i],
            "Holiday": holiday[i],
            "Special_Event": special_event[i],
            "Air_Quality_Index": round(aqi[i], 1),
            "Parking_Occupancy": round(parking_occ[i], 2),
            "Population_Density": int(pop_density[i]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# =============================================================================
# DATA LOADER
# =============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file=None):
    """Load CSV (uploaded or synthetic), clean, engineer features, return df."""
    is_synthetic = False
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalise column names to project schema
            col_map = {
                "id": "Record_ID", "timestamp": "Timestamp",
                "date": "Date", "time": "Time",
                "day_of_week": "Day_of_Week", "month": "Month",
                "hour": "Hour", "location": "Location",
                "road_type": "Road_Type", "vehicle_count": "Vehicle_Count",
                "average_speed": "Average_Speed_kmph",
                "traffic_density": "Traffic_Density",
                "travel_time": "Travel_Time_min",
                "weather": "Weather", "temperature": "Temperature_C",
                "rainfall": "Rainfall_mm", "visibility": "Visibility_km",
                "accident": "Accident_Occurred",
                "road_condition": "Road_Condition",
                "signal_status": "Traffic_Signal_Status",
                "parking_occupancy": "Parking_Occupancy",
                "population_density": "Population_Density",
                "congestion_level": "Congestion_Level",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            # Add missing columns expected by downstream code
            for col in ["Latitude","Longitude","Traffic_Volume","Year",
                        "Accident_Severity","Emergency_Response_Time",
                        "Holiday","Special_Event","Air_Quality_Index"]:
                if col not in df.columns:
                    df[col] = np.nan
            if "Traffic_Volume" not in df.columns or df["Traffic_Volume"].isna().all():
                df["Traffic_Volume"] = df.get("Vehicle_Count", 0)
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}. Using synthetic data.")
            df = generate_synthetic_data()
            is_synthetic = True
    else:
        df = generate_synthetic_data()
        is_synthetic = True

    # ── CLEANING ─────────────────────────────────────────────────────────────
    # Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Year" not in df.columns or df["Year"].isna().all():
        df["Year"] = df["Date"].dt.year if "Date" in df.columns else 2024

    # Duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    dups_removed = before - after

    # Missing values — numeric → median, categorical → mode
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].median(), inplace=True)
    for c in cat_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].mode()[0] if len(df[c].mode()) else "Unknown", inplace=True)

    # Outlier handling — IQR cap on numeric columns
    for c in ["Traffic_Volume", "Average_Speed_kmph", "Traffic_Density", "Temperature_C"]:
        if c in df.columns:
            Q1, Q3 = df[c].quantile(0.01), df[c].quantile(0.99)
            df[c] = df[c].clip(Q1, Q3)

    # ── FEATURE ENGINEERING ───────────────────────────────────────────────────
    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    if "Hour" in df.columns:
        df["Hour"] = df["Hour"].fillna(0).astype(int)
        df["Time_Period"] = pd.cut(
            df["Hour"],
            bins=[-1, 5, 9, 12, 17, 20, 23],
            labels=["Night (0-5)", "Morning Peak (6-9)", "Midday (10-12)",
                    "Afternoon (13-17)", "Evening Peak (18-20)", "Late Evening (21-23)"]
        )
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(int)
        df["Month_Name"] = pd.to_datetime(df["Month"], format="%m").dt.strftime("%b")
    if "Day_of_Week" in df.columns:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        df["Day_of_Week"] = pd.Categorical(df["Day_of_Week"], categories=day_order, ordered=True)
    # Risk score
    cong_map = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
    df["Congestion_Num"] = df["Congestion_Level"].map(cong_map).fillna(0).astype(int)
    df["Risk_Score"] = (
        df["Accident_Occurred"].astype(float) * 5 +
        df["Congestion_Num"] * 2 +
        ((df["Weather"] == "Heavy Rain") | (df["Weather"] == "Fog")).astype(float) +
        (df["Road_Condition"] == "Poor").astype(float)
    )

    df["dups_removed"] = dups_removed
    df["is_synthetic"] = is_synthetic
    return df

# =============================================================================
# KPI CARD HELPER
# =============================================================================
def kpi_card(icon, value, label):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <p class="kpi-value">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>"""

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>🚦 TrafficPulse AI</h2>
            <p>Traffic · Congestion · Accidents</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📂 Data Source")
        uploaded = st.file_uploader(
            "Upload CSV (optional)", type=["csv"],
            help="Upload your own traffic CSV or use the default dataset."
        )

        st.markdown("---")
        st.markdown("### 🔭 Navigation")
        page = st.radio("", [
            "🏠 Overview",
            "📊 Traffic Analysis",
            "🚨 Accident Analysis",
            "🌦️ Weather & Environment",
            "🗺️ Interactive Map",
            "🤖 ML Predictions",
            "📥 Download Data",
            "ℹ️ About Project",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        # Date range
        if "Date" in df.columns and df["Date"].notna().any():
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        else:
            date_range = None

        # Location
        locations = sorted(df["Location"].dropna().unique().tolist())
        sel_locations = st.multiselect("Location", locations, default=locations[:5])

        # Weather
        weathers = sorted(df["Weather"].dropna().unique().tolist())
        sel_weather = st.multiselect("Weather", weathers, default=weathers)

        # Road Type
        road_types = sorted(df["Road_Type"].dropna().unique().tolist())
        sel_road_type = st.multiselect("Road Type", road_types, default=road_types)

        # Road Condition
        road_conds = sorted(df["Road_Condition"].dropna().unique().tolist())
        sel_road_cond = st.multiselect("Road Condition", road_conds, default=road_conds)

        # Day of week
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        avail_days = [d for d in days if d in df["Day_of_Week"].unique().tolist()]
        sel_days = st.multiselect("Day of Week", avail_days, default=avail_days)

        # Congestion
        cong_levels = [c for c in ["Low","Medium","High","Severe"] if c in df["Congestion_Level"].unique()]
        sel_cong = st.multiselect("Congestion Level", cong_levels, default=cong_levels)

        # Accident severity
        sev_levels = sorted(df["Accident_Severity"].dropna().unique().tolist()) if "Accident_Severity" in df.columns else []
        sel_sev = st.multiselect("Accident Severity", sev_levels, default=sev_levels)

    return uploaded, page, date_range, sel_locations, sel_weather, sel_road_type, sel_road_cond, sel_days, sel_cong, sel_sev

# =============================================================================
# APPLY FILTERS
# =============================================================================
def apply_filters(df, date_range, sel_locations, sel_weather, sel_road_type,
                  sel_road_cond, sel_days, sel_cong, sel_sev):
    fdf = df.copy()
    if date_range and len(date_range) == 2 and "Date" in fdf.columns:
        fdf = fdf[(fdf["Date"].dt.date >= date_range[0]) & (fdf["Date"].dt.date <= date_range[1])]
    if sel_locations:
        fdf = fdf[fdf["Location"].isin(sel_locations)]
    if sel_weather:
        fdf = fdf[fdf["Weather"].isin(sel_weather)]
    if sel_road_type:
        fdf = fdf[fdf["Road_Type"].isin(sel_road_type)]
    if sel_road_cond:
        fdf = fdf[fdf["Road_Condition"].isin(sel_road_cond)]
    if sel_days:
        fdf = fdf[fdf["Day_of_Week"].isin(sel_days)]
    if sel_cong:
        fdf = fdf[fdf["Congestion_Level"].isin(sel_cong)]
    if sel_sev and "Accident_Severity" in fdf.columns:
        fdf = fdf[fdf["Accident_Severity"].isin(sel_sev)]
    return fdf

# =============================================================================
# PAGE: OVERVIEW (KPIs + summary)
# =============================================================================
def page_overview(fdf):
    st.markdown('<p class="section-header">📊 Key Performance Indicators</p>', unsafe_allow_html=True)
    total     = len(fdf)
    total_vol = int(fdf["Traffic_Volume"].sum()) if "Traffic_Volume" in fdf.columns else int(fdf["Vehicle_Count"].sum())
    accidents = int(fdf["Accident_Occurred"].sum()) if "Accident_Occurred" in fdf.columns else 0
    acc_rate  = f"{accidents/total*100:.2f}%" if total > 0 else "0%"
    avg_spd   = f"{fdf['Average_Speed_kmph'].mean():.1f}" if "Average_Speed_kmph" in fdf.columns else "N/A"
    avg_cong  = fdf["Congestion_Level"].mode()[0] if "Congestion_Level" in fdf.columns and len(fdf) else "N/A"
    high_risk = fdf.groupby("Location")["Accident_Occurred"].sum().idxmax() if total else "N/A"

    cols = st.columns(7)
    kpi_data = [
        ("🗂️", f"{total:,}",   "Total Records"),
        ("🚗", f"{total_vol:,}", "Total Traffic Volume"),
        ("🚨", f"{accidents:,}", "Accidents"),
        ("⚠️", acc_rate,         "Accident Rate"),
        ("🏎️", f"{avg_spd} km/h","Avg Speed"),
        ("🔴", avg_cong,         "Dominant Congestion"),
        ("📍", high_risk,        "Highest Risk Location"),
    ]
    for col, (icon, val, label) in zip(cols, kpi_data):
        with col:
            st.markdown(kpi_card(icon, val, label), unsafe_allow_html=True)

    st.markdown('<p class="section-header">📈 Quick Summary</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        # Traffic volume trend (daily)
        if "Date" in fdf.columns:
            daily = fdf.groupby("Date")["Traffic_Volume"].sum().reset_index()
            fig = px.line(daily, x="Date", y="Traffic_Volume", title="Daily Traffic Volume Trend",
                          color_discrete_sequence=["#4fc3f7"])
            st.plotly_chart(styled_fig(fig, 320), use_container_width=True)
    with c2:
        # Congestion donut
        cong_cnt = fdf["Congestion_Level"].value_counts().reset_index()
        cong_cnt.columns = ["Level", "Count"]
        fig = px.pie(cong_cnt, names="Level", values="Count", title="Congestion Distribution",
                     hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(styled_fig(fig, 320), use_container_width=True)
    with c3:
        # Accidents by location
        acc_loc = fdf.groupby("Location")["Accident_Occurred"].sum().sort_values(ascending=False).head(8).reset_index()
        fig = px.bar(acc_loc, x="Accident_Occurred", y="Location", orientation="h",
                     title="Top Accident Locations", color="Accident_Occurred",
                     color_continuous_scale="Reds")
        st.plotly_chart(styled_fig(fig, 320), use_container_width=True)

    # Automated insights
    st.markdown('<p class="section-header">💡 Automated Insights</p>', unsafe_allow_html=True)
    insights = generate_insights(fdf)
    for level, txt in insights:
        cls = f"alert-{level}"
        st.markdown(f'<div class="{cls}">💬 {txt}</div>', unsafe_allow_html=True)

# =============================================================================
# AUTOMATED INSIGHTS
# =============================================================================
def generate_insights(df):
    insights = []
    total = len(df)
    if total == 0:
        return [("low", "No data available for the current filter selection.")]

    # Accident rate
    acc_rate = df["Accident_Occurred"].mean() * 100 if "Accident_Occurred" in df.columns else 0
    if acc_rate > 3:
        insights.append(("high", f"Accident rate is {acc_rate:.2f}% — significantly above safe threshold. Immediate review recommended."))
    elif acc_rate > 1:
        insights.append(("medium", f"Accident rate stands at {acc_rate:.2f}%. Monitoring is advised."))
    else:
        insights.append(("low", f"Accident rate is low at {acc_rate:.2f}%. Traffic conditions appear relatively safe."))

    # Peak congestion hour
    if "Hour" in df.columns:
        peak_hr = df.groupby("Hour")["Congestion_Num"].mean().idxmax()
        insights.append(("medium", f"Peak congestion hour is {peak_hr:02d}:00. Consider signal optimisation during this window."))

    # Weather impact
    if "Weather" in df.columns and "Accident_Occurred" in df.columns:
        w_acc = df.groupby("Weather")["Accident_Occurred"].mean()
        worst_w = w_acc.idxmax()
        insights.append(("high", f"'{worst_w}' weather has the highest accident probability ({w_acc[worst_w]*100:.1f}%). Enhanced warnings should be activated."))

    # High congestion share
    high_cong_pct = (df["Congestion_Level"].isin(["High","Severe"])).mean() * 100
    if high_cong_pct > 20:
        insights.append(("high", f"{high_cong_pct:.1f}% of records show High/Severe congestion — infrastructure intervention may be needed."))
    else:
        insights.append(("low", f"High/Severe congestion accounts for {high_cong_pct:.1f}% of observations."))

    # Speed insight
    avg_spd = df["Average_Speed_kmph"].mean() if "Average_Speed_kmph" in df.columns else None
    if avg_spd is not None:
        insights.append(("low" if avg_spd > 30 else "medium",
                         f"Average vehicle speed is {avg_spd:.1f} km/h — {'adequate flow' if avg_spd > 30 else 'indicative of congestion'}."))

    return insights

# =============================================================================
# PAGE: TRAFFIC ANALYSIS
# =============================================================================
def page_traffic(fdf):
    st.markdown('<p class="section-header">📊 Traffic Volume & Flow Analysis</p>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Volume Trends", "⏰ Peak Hours", "📆 Day & Month", "🔗 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if "Date" in fdf.columns:
                daily = fdf.groupby("Date")["Traffic_Volume"].sum().reset_index()
                fig = px.area(daily, x="Date", y="Traffic_Volume", title="Daily Traffic Volume",
                              color_discrete_sequence=["#4fc3f7"])
                fig.update_traces(fill="tozeroy", fillcolor="rgba(79,195,247,0.15)")
                st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            loc_vol = fdf.groupby("Location")["Traffic_Volume"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(loc_vol, x="Location", y="Traffic_Volume", title="Avg Traffic Volume by Location",
                         color="Traffic_Volume", color_continuous_scale="Blues")
            st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if "Hour" in fdf.columns:
                hourly = fdf.groupby("Hour")["Traffic_Volume"].mean().reset_index()
                fig = px.bar(hourly, x="Hour", y="Traffic_Volume", title="Average Traffic Volume by Hour",
                             color="Traffic_Volume", color_continuous_scale="Viridis")
                st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            if "Hour" in fdf.columns:
                heatmap_data = fdf.groupby(["Day_of_Week","Hour"])["Traffic_Volume"].mean().unstack(fill_value=0)
                fig = px.imshow(heatmap_data, title="Traffic Heatmap: Day vs Hour",
                                color_continuous_scale="Blues", aspect="auto")
                st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            day_vol = fdf.groupby("Day_of_Week")["Traffic_Volume"].mean().reset_index()
            fig = px.bar(day_vol, x="Day_of_Week", y="Traffic_Volume",
                         title="Avg Traffic Volume by Day", color="Traffic_Volume",
                         color_continuous_scale="Purples")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            month_vol = fdf.groupby("Month")["Traffic_Volume"].sum().reset_index()
            fig = px.line(month_vol, x="Month", y="Traffic_Volume",
                          title="Monthly Total Traffic Volume", markers=True,
                          color_discrete_sequence=["#bb86fc"])
            st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            # Speed vs Congestion
            fig = px.box(fdf, x="Congestion_Level", y="Average_Speed_kmph",
                         title="Speed vs Congestion Level",
                         color="Congestion_Level",
                         category_orders={"Congestion_Level": ["Low","Medium","High","Severe"]},
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            # Volume vs Accidents
            sample = fdf.sample(min(2000, len(fdf)), random_state=42)
            fig = px.scatter(sample, x="Traffic_Volume", y="Average_Speed_kmph",
                             color="Congestion_Level", size_max=8,
                             title="Traffic Volume vs Speed",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             opacity=0.6)
            st.plotly_chart(styled_fig(fig), use_container_width=True)

        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        num_df = fdf.select_dtypes(include=[np.number]).drop(
            columns=["Record_ID","dups_removed","is_synthetic"], errors="ignore"
        ).dropna()
        if len(num_df.columns) >= 3:
            corr = num_df.corr()
            fig = px.imshow(corr, title="Feature Correlation Heatmap",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
            st.plotly_chart(styled_fig(fig, 550), use_container_width=True)

        # Vehicle type distribution (if available)
        veh_cols = [c for c in ["Car_Count","Bike_Count","Bus_Count","Truck_Count","Pedestrian_Count"] if c in fdf.columns]
        if veh_cols:
            veh_totals = fdf[veh_cols].sum().reset_index()
            veh_totals.columns = ["Vehicle Type","Count"]
            fig = px.pie(veh_totals, names="Vehicle Type", values="Count",
                         title="Vehicle Type Distribution", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

# =============================================================================
# PAGE: ACCIDENT ANALYSIS
# =============================================================================
def page_accidents(fdf):
    st.markdown('<p class="section-header">🚨 Accident Analysis</p>', unsafe_allow_html=True)
    acc_df = fdf[fdf["Accident_Occurred"] == 1] if "Accident_Occurred" in fdf.columns else fdf.head(0)
    total_acc = len(acc_df)
    st.info(f"📋 **{total_acc}** accidents found in the current filtered dataset.")

    tab1, tab2, tab3 = st.tabs(["🌦️ By Weather & Road", "📍 By Location", "⏱️ By Time & Severity"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            w_acc = fdf.groupby("Weather")["Accident_Occurred"].agg(["sum","count"])
            w_acc["Rate"] = w_acc["sum"] / w_acc["count"] * 100
            w_acc = w_acc.sort_values("Rate", ascending=False).reset_index()
            fig = px.bar(w_acc, x="Weather", y="Rate", title="Accident Rate by Weather (%)",
                         color="Rate", color_continuous_scale="Reds")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            rc_acc = fdf.groupby("Road_Condition")["Accident_Occurred"].agg(["sum","count"])
            rc_acc["Rate"] = rc_acc["sum"] / rc_acc["count"] * 100
            rc_acc = rc_acc.sort_values("Rate", ascending=False).reset_index()
            fig = px.bar(rc_acc, x="Road_Condition", y="Rate",
                         title="Accident Rate by Road Condition (%)",
                         color="Rate", color_continuous_scale="Oranges")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            rt_acc = fdf.groupby("Road_Type")["Accident_Occurred"].mean().mul(100).reset_index()
            fig = px.bar(rt_acc, x="Road_Type", y="Accident_Occurred",
                         title="Accident Rate by Road Type (%)",
                         color="Accident_Occurred", color_continuous_scale="YlOrRd",
                         labels={"Accident_Occurred": "Rate (%)"})
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c4:
            if "Accident_Severity" in fdf.columns:
                sev = acc_df["Accident_Severity"].value_counts().reset_index()
                sev.columns = ["Severity","Count"]
                fig = px.pie(sev, names="Severity", values="Count",
                             title="Accident Severity Distribution",
                             color_discrete_sequence=["#27ae60","#f39c12","#e74c3c"])
                st.plotly_chart(styled_fig(fig), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            loc_acc = fdf.groupby("Location")["Accident_Occurred"].sum().sort_values(ascending=True).reset_index()
            fig = px.bar(loc_acc, x="Accident_Occurred", y="Location", orientation="h",
                         title="Total Accidents by Location",
                         color="Accident_Occurred", color_continuous_scale="Reds")
            st.plotly_chart(styled_fig(fig, 450), use_container_width=True)
        with c2:
            loc_rate = fdf.groupby("Location")["Accident_Occurred"].mean().mul(100).sort_values(ascending=True).reset_index()
            fig = px.bar(loc_rate, x="Accident_Occurred", y="Location", orientation="h",
                         title="Accident Rate by Location (%)",
                         color="Accident_Occurred", color_continuous_scale="Oranges",
                         labels={"Accident_Occurred": "Rate (%)"})
            st.plotly_chart(styled_fig(fig, 450), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if "Hour" in fdf.columns:
                hr_acc = fdf.groupby("Hour")["Accident_Occurred"].sum().reset_index()
                fig = px.bar(hr_acc, x="Hour", y="Accident_Occurred",
                             title="Accidents by Hour of Day",
                             color="Accident_Occurred", color_continuous_scale="Reds")
                st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c2:
            day_acc = fdf.groupby("Day_of_Week")["Accident_Occurred"].sum().reset_index()
            fig = px.bar(day_acc, x="Day_of_Week", y="Accident_Occurred",
                         title="Accidents by Day of Week",
                         color="Accident_Occurred", color_continuous_scale="Oranges")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        # ERT if available
        if "Emergency_Response_Time" in fdf.columns:
            ert = acc_df["Emergency_Response_Time"].dropna()
            if len(ert) > 5:
                fig = px.histogram(ert, title="Emergency Response Time Distribution (minutes)",
                                   nbins=30, color_discrete_sequence=["#4fc3f7"])
                st.plotly_chart(styled_fig(fig), use_container_width=True)

# =============================================================================
# PAGE: WEATHER & ENVIRONMENT
# =============================================================================
def page_weather(fdf):
    st.markdown('<p class="section-header">🌦️ Weather & Environmental Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        w_cong = fdf.groupby(["Weather","Congestion_Level"]).size().reset_index(name="Count")
        fig = px.bar(w_cong, x="Weather", y="Count", color="Congestion_Level",
                     title="Weather vs Congestion Level",
                     color_discrete_sequence=px.colors.sequential.Blues_r,
                     barmode="stack")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
    with c2:
        fig = px.scatter(fdf.sample(min(3000,len(fdf)), random_state=1),
                         x="Visibility_km", y="Average_Speed_kmph",
                         color="Weather", title="Visibility vs Average Speed",
                         opacity=0.5, color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        temp_cong = fdf.groupby("Congestion_Level")["Temperature_C"].mean().reset_index()
        fig = px.bar(temp_cong, x="Congestion_Level", y="Temperature_C",
                     title="Avg Temperature by Congestion Level",
                     color="Temperature_C", color_continuous_scale="RdYlGn_r",
                     category_orders={"Congestion_Level": ["Low","Medium","High","Severe"]})
        st.plotly_chart(styled_fig(fig), use_container_width=True)
    with c4:
        if "Air_Quality_Index" in fdf.columns and fdf["Air_Quality_Index"].notna().any():
            fig = px.box(fdf, x="Congestion_Level", y="Air_Quality_Index",
                         title="Air Quality Index by Congestion Level",
                         color="Congestion_Level",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         category_orders={"Congestion_Level": ["Low","Medium","High","Severe"]})
            st.plotly_chart(styled_fig(fig), use_container_width=True)

    # Rainfall impact
    if "Rainfall_mm" in fdf.columns and fdf["Rainfall_mm"].notna().any():
        c5, c6 = st.columns(2)
        with c5:
            fig = px.scatter(fdf.sample(min(2000,len(fdf)), random_state=2),
                             x="Rainfall_mm", y="Traffic_Volume", color="Congestion_Level",
                             title="Rainfall vs Traffic Volume",
                             color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.5)
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with c6:
            fig = px.histogram(fdf, x="Temperature_C", nbins=40,
                               color="Congestion_Level",
                               title="Temperature Distribution by Congestion",
                               color_discrete_sequence=px.colors.qualitative.Pastel,
                               barmode="overlay", opacity=0.7)
            st.plotly_chart(styled_fig(fig), use_container_width=True)

# =============================================================================
# PAGE: INTERACTIVE MAP
# =============================================================================
def page_map(fdf):
    st.markdown('<p class="section-header">🗺️ Interactive Traffic Map</p>', unsafe_allow_html=True)
    if "Latitude" not in fdf.columns or fdf["Latitude"].isna().all():
        st.warning("⚠️ No geographic coordinates found in the dataset. Map is unavailable.")
        return

    map_df = fdf.dropna(subset=["Latitude","Longitude"])
    if len(map_df) == 0:
        st.warning("⚠️ No valid coordinate data after filtering.")
        return

    # Aggregate by location for cleaner map
    agg = map_df.groupby(["Location","Latitude","Longitude"]).agg(
        Accidents=("Accident_Occurred", "sum"),
        Avg_Speed=("Average_Speed_kmph", "mean"),
        Avg_Volume=("Traffic_Volume", "mean"),
        Count=("Traffic_Volume", "count"),
        Dominant_Congestion=("Congestion_Level", lambda x: x.mode()[0]),
    ).reset_index()

    col1, col2 = st.columns([3, 1])
    with col2:
        map_metric = st.selectbox("Colour by", ["Accidents","Avg_Speed","Avg_Volume","Count"])
        map_type   = st.selectbox("Map Type", ["Scatter", "Density"])
    with col1:
        if map_type == "Scatter":
            fig = px.scatter_mapbox(
                agg, lat="Latitude", lon="Longitude",
                size="Count", color=map_metric, hover_name="Location",
                hover_data={"Accidents": True, "Avg_Speed": ":.1f", "Avg_Volume": ":.0f",
                            "Dominant_Congestion": True},
                color_continuous_scale="Reds" if map_metric=="Accidents" else "Blues",
                zoom=11, height=560, mapbox_style="carto-darkmatter",
                title=f"Traffic Map — {map_metric}"
            )
        else:
            fig = px.density_mapbox(
                map_df.sample(min(5000,len(map_df)), random_state=5),
                lat="Latitude", lon="Longitude", z="Traffic_Volume",
                radius=18, zoom=11, height=560,
                mapbox_style="carto-darkmatter",
                color_continuous_scale="Viridis",
                title="Traffic Density Heatmap"
            )
        fig.update_layout(template=PLOTLY_THEME, paper_bgcolor=CHART_BG, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: ML PREDICTIONS
# =============================================================================
def page_ml(fdf):
    st.markdown('<p class="section-header">🤖 Machine Learning Predictions</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    Two ML models are available:<br>
    <b>1. Congestion Level Classifier</b> — Predicts Low / Medium / High / Severe congestion from traffic features.<br>
    <b>2. Accident Occurrence Classifier</b> — Predicts whether an accident is likely to occur.
    </div>""", unsafe_allow_html=True)

    task = st.radio("Select Prediction Task", ["Congestion Level", "Accident Occurrence"], horizontal=True)
    algo = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    test_size = st.slider("Test Split (%)", 10, 40, 20) / 100

    if st.button("🚀 Train & Evaluate Model", use_container_width=True):
        with st.spinner("Training model…"):
            result = train_model(fdf, task, algo, test_size)
        if result is None:
            st.error("Insufficient data to train the model with current filters.")
            return
        metrics, cm, report_text, feat_imp, classes = result

        st.markdown("#### 📊 Model Performance Metrics")
        cols = st.columns(len(metrics))
        metric_icons = {"Accuracy": "🎯", "Precision": "🔍", "Recall": "📡", "F1 Score": "⚖️", "ROC-AUC": "📈"}
        for col, (name, val) in zip(cols, metrics.items()):
            with col:
                icon = metric_icons.get(name, "📊")
                st.markdown(kpi_card(icon, f"{val:.3f}", name), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🧩 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, linewidths=0.5, linecolor="#2e3450",
                        annot_kws={"size": 12, "color": "white"})
            ax.set_xlabel("Predicted", color="#ccd6f6")
            ax.set_ylabel("Actual", color="#ccd6f6")
            ax.set_title("Confusion Matrix", color="#ccd6f6")
            ax.tick_params(colors="#ccd6f6")
            st.pyplot(fig)
            plt.close()

        with c2:
            if feat_imp is not None:
                st.markdown("#### 🔑 Feature Importance (Top 12)")
                feat_df = feat_imp.head(12)
                fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues",
                             title="Top Feature Importances")
                st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

        with st.expander("📋 Full Classification Report"):
            st.code(report_text, language="text")

@st.cache_data(show_spinner=False)
def train_model(df, task, algo, test_size):
    """Train and evaluate ML model. Returns metrics, confusion matrix, report, feature importance."""
    feature_cols = ["Hour", "Month", "Traffic_Volume", "Average_Speed_kmph",
                    "Traffic_Density", "Temperature_C", "Visibility_km",
                    "Congestion_Num", "Population_Density", "Rainfall_mm",
                    "Parking_Occupancy", "Travel_Time_min"]
    cat_features = ["Weather", "Road_Type", "Road_Condition",
                    "Traffic_Signal_Status", "Day_of_Week"]

    # Build working dataframe
    avail_num = [c for c in feature_cols if c in df.columns]
    avail_cat = [c for c in cat_features if c in df.columns]

    wdf = df[avail_num + avail_cat].copy()

    # Encode categoricals
    le_map = {}
    for c in avail_cat:
        le = LabelEncoder()
        wdf[c] = le.fit_transform(wdf[c].astype(str))
        le_map[c] = le

    if task == "Congestion Level":
        if "Congestion_Level" not in df.columns:
            return None
        y = LabelEncoder().fit_transform(df["Congestion_Level"].astype(str))
        le_y = LabelEncoder()
        le_y.fit(df["Congestion_Level"].astype(str))
        classes = le_y.classes_
        # Drop Congestion_Num from features to avoid leakage
        if "Congestion_Num" in wdf.columns:
            wdf = wdf.drop(columns=["Congestion_Num"])
    else:
        if "Accident_Occurred" not in df.columns:
            return None
        y = df["Accident_Occurred"].astype(int).values
        classes = np.array(["No Accident","Accident"])

    X = wdf.fillna(wdf.median(numeric_only=True))
    if len(X) < 50:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if algo == "Random Forest":
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif algo == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=500, random_state=42)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        if y_prob is not None and len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        elif y_prob is not None:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "ROC-AUC": auc}
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    # Feature importance
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

    return metrics, cm, report_text, feat_imp, classes

# =============================================================================
# PAGE: DOWNLOAD
# =============================================================================
def page_download(fdf, df):
    st.markdown('<p class="section-header">📥 Download Data & Reports</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📋 Filtered Dataset")
        st.dataframe(fdf.drop(columns=["dups_removed","is_synthetic"], errors="ignore").head(1000), height=300)
        csv = fdf.drop(columns=["dups_removed","is_synthetic"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered CSV", csv, "TrafficPulse_filtered.csv", "text/csv")
    with c2:
        st.markdown("#### 📊 Analysis Summary")
        summary = {
            "Total Records": len(fdf),
            "Total Accidents": int(fdf["Accident_Occurred"].sum()),
            "Accident Rate (%)": round(fdf["Accident_Occurred"].mean()*100, 3),
            "Avg Speed (kmph)": round(fdf["Average_Speed_kmph"].mean(), 2),
            "High Congestion %": round((fdf["Congestion_Level"].isin(["High","Severe"])).mean()*100, 2),
            "Total Traffic Volume": int(fdf["Traffic_Volume"].sum()),
        }
        sum_df = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
        st.dataframe(sum_df, height=260)
        sum_csv = sum_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Summary CSV", sum_csv, "TrafficPulse_summary.csv", "text/csv")

    st.markdown("#### 📈 Full Dataset Statistics")
    st.dataframe(fdf.select_dtypes(include=[np.number]).describe().round(2), height=280)

# =============================================================================
# PAGE: ABOUT
# =============================================================================
def page_about():
    st.markdown('<p class="section-header">ℹ️ About TrafficPulse AI</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <h3 style="color:#4fc3f7; margin-top:0;">🚦 TrafficPulse AI — Traffic, Congestion & Accident Analytics</h3>
    <p><b>Version:</b> 1.0 &nbsp;|&nbsp; <b>Author:</b> Amit Dutta &nbsp;|&nbsp; <b>Report Author:</b> Rajshekhar Jana</p>
    <hr style="border-color:#2e3450;">

    <h4 style="color:#ccd6f6;">🎯 Objectives</h4>
    <ul>
        <li>Provide actionable insights from urban traffic and accident data</li>
        <li>Identify congestion hotspots, peak hours, and high-risk locations</li>
        <li>Analyse the impact of weather, road condition, and time on traffic patterns</li>
        <li>Enable data-driven decision making for traffic management authorities</li>
        <li>Predict congestion levels and accident occurrences using machine learning</li>
    </ul>

    <h4 style="color:#ccd6f6;">📦 Dataset</h4>
    <p>The application accepts a user-uploaded CSV (DOC-20260921-WA0011.csv or any compatible traffic CSV).
    If no file is uploaded, a realistic synthetic dataset is automatically generated using statistically
    calibrated random distributions that mimic real Kolkata traffic patterns.</p>

    <h4 style="color:#ccd6f6;">🔬 Methodology</h4>
    <ol>
        <li><b>Data Loading & Validation</b> — schema normalisation, type coercion</li>
        <li><b>Cleaning</b> — duplicate removal, missing value imputation (median/mode), IQR outlier capping</li>
        <li><b>Feature Engineering</b> — time period bins, risk score, congestion numeric encoding</li>
        <li><b>EDA & Statistics</b> — volume trends, peak analysis, correlation heatmaps</li>
        <li><b>ML Modelling</b> — Random Forest, Gradient Boosting, Logistic Regression with stratified split</li>
        <li><b>Evaluation</b> — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix</li>
    </ol>

    <h4 style="color:#ccd6f6;">🛠️ Technologies Used</h4>
    <ul>
        <li>Python 3.10+, Streamlit 1.35+</li>
        <li>Pandas, NumPy — data manipulation</li>
        <li>Plotly Express / Graph Objects — interactive charts & maps</li>
        <li>Matplotlib, Seaborn — static visualisations</li>
        <li>Scikit-learn — ML pipelines, metrics</li>
    </ul>

    <h4 style="color:#ccd6f6;">⚠️ Limitations</h4>
    <ul>
        <li>Synthetic data may not perfectly reflect real-world traffic dynamics</li>
        <li>ML models are trained on a single dataset; generalisability is limited</li>
        <li>Real-time data feed is not implemented</li>
        <li>GPS coordinates are approximate cluster centres for Kolkata localities</li>
    </ul>

    <h4 style="color:#ccd6f6;">🚀 How to Run</h4>
    <pre style="background:#0f1117; padding:12px; border-radius:6px; color:#4fc3f7;">
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run "Amit Dutta_TrafficPulseAI.py"</pre>
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Render sidebar and get controls
    # We need a temporary load to get filter options; actual load happens after upload detection
    temp_df = generate_synthetic_data()
    uploaded, page, date_range, sel_locations, sel_weather, sel_road_type, \
        sel_road_cond, sel_days, sel_cong, sel_sev = render_sidebar(temp_df)

    # Load real data based on upload
    with st.spinner("⏳ Loading and processing data…"):
        df = load_and_clean(uploaded)

    is_syn = bool(df["is_synthetic"].iloc[0]) if "is_synthetic" in df.columns else False
    dups   = int(df["dups_removed"].iloc[0]) if "dups_removed" in df.columns else 0

    # Header
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1 style="font-size:2.4rem; color:#4fc3f7; margin:0;">
            🚦 TrafficPulse AI
        </h1>
        <p style="color:#8892b0; font-size:1.05rem; margin:6px 0 0 0;">
            Traffic · Congestion · Accident Analytics Dashboard
        </p>
    </div>""", unsafe_allow_html=True)

    if is_syn:
        st.warning("⚠️ **Synthetic Dataset Active** — Upload your real CSV via the sidebar to analyse actual traffic data. "
                   "All visualisations and ML results are based on simulated data.")
    else:
        st.success(f"✅ **Dataset Loaded** — {len(df):,} records. Duplicates removed: {dups}. "
                   "All analyses reflect the uploaded real dataset.")

    # Apply filters
    fdf = apply_filters(df, date_range, sel_locations, sel_weather, sel_road_type,
                        sel_road_cond, sel_days, sel_cong, sel_sev)
    if len(fdf) == 0:
        st.error("❌ No data matches the selected filters. Please adjust the filter criteria.")
        st.stop()

    st.caption(f"📊 Showing **{len(fdf):,}** records after applying filters (out of {len(df):,} total)")

    # Route to page
    if "Overview"      in page: page_overview(fdf)
    elif "Traffic"     in page: page_traffic(fdf)
    elif "Accident"    in page: page_accidents(fdf)
    elif "Weather"     in page: page_weather(fdf)
    elif "Map"         in page: page_map(fdf)
    elif "ML"          in page: page_ml(fdf)
    elif "Download"    in page: page_download(fdf, df)
    elif "About"       in page: page_about()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#57606a; font-size:0.8rem;'>"
        "TrafficPulse AI &nbsp;|&nbsp; Amit Dutta &nbsp;|&nbsp; Report: Rajshekhar Jana &nbsp;|&nbsp; "
        "Data is synthetic/illustrative unless a real CSV is uploaded."
        "</p>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
