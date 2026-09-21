# 🚦 TrafficPulse AI — Traffic, Congestion & Accident Analytics

> **A complete, professional AI-powered traffic analytics dashboard built with Python and Streamlit.**

---

## 📋 Project Description

**TrafficPulse AI** is an end-to-end Data Analysis and Machine Learning project that transforms raw urban traffic data into actionable intelligence. The dashboard provides deep insights into traffic volume patterns, congestion hotspots, accident risk factors, and environmental impacts — all through a visually rich, interactive Streamlit interface.

Designed for traffic management authorities, urban planners, data science students, and infrastructure analysts, this project covers the complete data science pipeline: ingestion → cleaning → EDA → visualisation → ML modelling → reporting.

---

## ✨ Key Features

- **🔄 Flexible Data Input** — Upload any compatible traffic CSV or auto-generate a realistic 5,000-row synthetic dataset
- **🧹 Automated Data Quality** — Duplicate removal, median/mode imputation, IQR outlier capping
- **⚙️ Feature Engineering** — Time-period bins, risk scores, congestion encoding
- **📊 Rich EDA** — 15+ interactive Plotly charts covering every dimension of traffic data
- **🗺️ Interactive Map** — Mapbox scatter and density heatmap of traffic hotspots
- **🤖 ML Predictions** — Congestion level & accident occurrence classifiers with 3 algorithms
- **📈 Model Metrics** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Feature Importance
- **💡 Auto-Insights** — Plain-language analytical insights generated dynamically
- **⬇️ CSV Downloads** — Filtered data and summary statistics export
- **🎨 Modern Dark UI** — Custom CSS with cards, tabs, KPI tiles, and responsive layout

---

## 🎯 Objectives

1. Identify traffic congestion patterns across time, location, weather, and road conditions
2. Detect high-risk accident zones and contributing environmental factors
3. Predict congestion levels and accident likelihood using supervised ML
4. Provide an intuitive, professional dashboard suitable for real-world deployment

---

## 📂 Dataset Description

### Primary Dataset: `DOC-20260921-WA0011.csv`
| Field | Description |
|-------|-------------|
| `id` | Unique record identifier |
| `timestamp` | Full datetime of observation |
| `date` | Date (YYYY-MM-DD) |
| `time` | Time (HH:MM:SS) |
| `day_of_week` | Day name (Monday–Sunday) |
| `month` | Month number (1–12) |
| `hour` | Hour of day (0–23) |
| `location` | Locality name in Kolkata |
| `road_type` | Arterial / Highway / Collector / Local |
| `vehicle_count` | Number of vehicles observed |
| `average_speed` | Mean speed in km/h |
| `traffic_density` | Vehicles per unit area |
| `travel_time` | Travel time in minutes |
| `weather` | Clear / Rain / Fog / Heavy Rain / Cloudy |
| `temperature` | Temperature in °C |
| `rainfall` | Rainfall in mm |
| `visibility` | Visibility in km |
| `accident` | Binary accident flag (0/1) |
| `road_condition` | Good / Fair / Poor / Wet |
| `signal_status` | Normal / Disrupted / Off |
| `parking_occupancy` | % parking spaces occupied |
| `population_density` | People per sq km |
| `congestion_level` | Low / Medium / High / Severe |

**Rows:** 12,000 · **Locations:** 12 Kolkata localities · **Date range:** Jan 2024 – May 2025

### Synthetic Fallback Dataset
If no file is uploaded, the app generates 5,000 rows with statistically calibrated distributions matching real Kolkata traffic behaviour. All synthetic outputs are clearly labelled.

---

## 🛠️ Technologies Used

| Category | Library/Tool |
|----------|-------------|
| Frontend | Streamlit 1.35+ |
| Data Manipulation | Pandas 2.0+, NumPy 1.24+ |
| Visualisation | Plotly 5.18+, Matplotlib 3.7+, Seaborn 0.12+ |
| Machine Learning | Scikit-learn 1.3+ |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
TrafficPulse-AI/
│
├── Amit Dutta_TrafficPulseAI.py          # Main application (all-in-one)
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── Amit Dutta_TrafficPulseAI_ProjectReport.docx  # Full project report
└── DOC-20260921-WA0011.csv               # Source dataset (upload via UI)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or newer
- pip package manager

### Steps

```bash
# 1. Clone or download the project
git clone <repository-url>
cd TrafficPulse-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run the dashboard
python -m streamlit run "Amit Dutta_TrafficPulseAI.py"
```

The dashboard will open automatically at **http://localhost:8501**

---

## 🖥️ Dashboard Features

### 🏠 Overview Page
- 7 KPI cards: Total Records, Traffic Volume, Accidents, Accident Rate, Avg Speed, Dominant Congestion, Highest Risk Location
- Daily traffic volume area chart
- Congestion level donut chart
- Top accident locations bar chart
- Auto-generated plain-language insights

### 📊 Traffic Analysis Page
- Daily volume area trend
- Average volume by location
- Hourly traffic bar chart
- Day × Hour traffic heatmap
- Day of week and monthly volume charts
- Speed vs congestion box plots
- Volume vs speed scatter
- Feature correlation heatmap
- Vehicle type distribution (if available)

### 🚨 Accident Analysis Page
- Accident rate by weather (%)
- Accident rate by road condition (%)
- Accident rate by road type (%)
- Accident severity pie chart
- Total accidents by location (horizontal bar)
- Accidents by hour and day of week
- Emergency response time histogram

### 🌦️ Weather & Environment Page
- Weather vs congestion stacked bar
- Visibility vs average speed scatter
- Temperature by congestion level
- Air quality index by congestion level
- Rainfall vs traffic volume
- Temperature distribution by congestion

### 🗺️ Interactive Map Page
- Mapbox scatter map coloured by accidents / speed / volume
- Mapbox density heatmap
- Toggle between map types and colour metrics

---

## 🤖 ML Features

| Task | Target | Algorithms |
|------|--------|-----------|
| Congestion Classification | Low/Medium/High/Severe | Random Forest, Gradient Boosting, Logistic Regression |
| Accident Occurrence | 0 (No) / 1 (Yes) | Random Forest, Gradient Boosting, Logistic Regression |

**Input features used:**
- Hour, Month, Traffic Volume, Average Speed, Traffic Density
- Temperature, Visibility, Congestion (numeric), Population Density
- Rainfall, Parking Occupancy, Travel Time
- Weather, Road Type, Road Condition, Signal Status, Day of Week (encoded)

**Evaluation outputs:**
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix (Seaborn heatmap)
- Feature Importance bar chart (tree-based models)
- Full classification report

---

## 📤 Expected Output

When you run the app, you will see:

1. A dark-themed Streamlit dashboard at `http://localhost:8501`
2. A sidebar with file upload + navigation + 8 interactive filters
3. KPI tiles updating dynamically based on filter selection
4. 15+ Plotly charts across 6 analysis pages
5. An interactive Mapbox map (requires internet for tiles)
6. A one-click ML training workflow with live metrics
7. CSV download buttons for filtered data and summaries

---

## ⚠️ Limitations

- Synthetic data is generated statistically and may not perfectly mirror actual traffic behaviour
- ML models are trained on a single dataset snapshot — real-time retraining is not implemented
- GPS coordinates are approximate centre-points for Kolkata localities (not exact GPS traces)
- Mapbox tiles require an internet connection
- Very large datasets (>500K rows) may cause UI slowdown without sampling

---

## 🔮 Future Improvements

- [ ] Real-time data ingestion via API (transport department feeds)
- [ ] Time-series forecasting (LSTM / Prophet) for traffic volume prediction
- [ ] Multi-city support with dynamic location geocoding
- [ ] Automated PDF report generation
- [ ] User authentication and role-based access
- [ ] Docker containerisation for easy deployment
- [ ] Integration with Google Maps / HERE Traffic API

---

## 👤 Author

**Rajshekhar Jana**
Data Science & AI Internship Project
Department of Computer Science & Engineering

---

## 📄 License

This project is licensed for academic and educational use. Not intended for production deployment without further validation against live traffic data.

---

*TrafficPulse AI · Version 1.0 · Built with ❤️ using Streamlit, Pandas, Plotly & Scikit-learn*
