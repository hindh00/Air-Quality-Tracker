
# 🌍 Qatar Air Quality AI Forecaster (Group 8)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive AI system designed to predict PM2.5 air quality levels 24 hours in advance for major municipalities in Qatar. This project leverages Deep Learning (Bidirectional LSTM) and real-time meteorological data to provide actionable health insights.

---

## 📖 Project Overview

Air pollution, specifically **PM2.5** (fine particulate matter), is a critical health challenge in the Arabian Gulf due to a mix of industrial emissions and natural dust storms.

This project solves the lack of predictive insight in current monitoring systems by:
1.  **Aggregating Data:** Fusing historical weather data with satellite-derived pollutant markers (Open-Meteo API).
2.  **Forecasting:** Using a **Universal Bidirectional LSTM** model to predict air quality 24 hours into the future.
3.  **Visualizing:** Presenting data via an interactive **Streamlit Dashboard** with 3D mapping.

### Key Features
* **Multi-Region Support:** Covers Doha, Al Khor, Al Rayyan, Al Wakrah, Umm Slal Ali, and Qatar General.
* **Simulation Mode:** Validates accuracy by using 48-hour old data to predict "yesterday's" air quality against known ground truth.
* **Interactive Map:** 3D PyDeck visualizations with color-coded EPA health zones.
* **Model Comparison:** Benchmarks against XGBoost and SARIMA.

---

## 📂 Repository Structure

```text
├── Dashboard/                  # The Deployment Application
│   ├── app.py                  # Main Streamlit application entry point
│   ├── requirements.txt        # Python dependencies for the dashboard
│   ├── Dockerfile              # Containerization setup
│   └── universal_lstm_model.keras # The deployed LSTM model file
│
├── Data/                       # Datasets
│   ├── final_training_data...  # Historical data used for model training
│   └── final_test_data...      # Data used for validation metrics
│
├── Models/                     # Research & Development
│   ├── LSTMModel3.0/           # THE FINAL MODEL ARCHITECTURE
│   │   ├── LSTM_V3.ipynb       # Training notebook
│   │   ├── lstm_model.keras    # Saved model artifact
│   │   └── scaler_X.pkl        # Feature scalers
│   ├── XGBoostModel/           # Comparative ML Model (High R², low temporal awareness)
│   └── SARIMAModel/            # Statistical Baseline
│
├── Presentation/               # Project Slides (Group 8).pptx
├── Report/                     # Final PDF Report
└── README.md                   # Project Documentation
````

-----

## 🚀 Installation & Setup

### Prerequisites

  * Python 3.9 or higher
  * Git

### 1\. Clone the Repository

```bash
git clone [https://github.com/your-username/Qatar-Air-Quality-Tracker.git](https://github.com/your-username/Qatar-Air-Quality-Tracker.git)
cd Qatar-Air-Quality-Tracker
```

### 2\. Run the Dashboard (Local)

Navigate to the Dashboard folder and install dependencies:

```bash
cd Dashboard
# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3\. Run with Docker (Optional)

If you prefer containerization:

```bash
cd Dashboard
docker build -t qatar-air-quality .
docker run -p 8501:8501 qatar-air-quality
```

-----

## 🧠 Model Performance

We evaluated three architectures. The **Bidirectional LSTM** was selected for production due to its superior ability to capture temporal dependencies (e.g., the onset of dust storms).

| Model Architecture | R² Score | RMSE (μg/m³) | Notes |
| :--- | :---: | :---: | :--- |
| **Bidirectional LSTM (Final)** | **0.9217** | **2.65** | Best balance of accuracy & temporal continuity. |
| XGBoost | 0.9277 | 3.66 | High accuracy but treats time steps independently. |
| SARIMA | \~0.75 | \~7.20 | Failed to capture non-linear weather spikes. |

### Regional Accuracy (LSTM)

  * **Doha:** 93.3%
  * **Al Wakrah:** 93.2%
  * **Al Rayyan:** 93.2%
  * **Al Khor:** 90.9%

-----

## 📊 Dashboard Usage Guide

1.  **Select Region:** Use the sidebar to choose a municipality (e.g., Doha).
2.  **View Forecast:** The map shows a color-coded zone.
      * 🟢 **Green:** Good
      * 🟡 **Yellow:** Moderate
      * 🔴 **Red:** Unhealthy
3.  **Time Slider:** Drag the slider to see how the pollution cloud moves over the next 24 hours.
4.  **Simulation Mode:** *Note: The dashboard currently simulates "Live" mode by lagging 2 days behind. This allows us to fetch historical weather data that is guaranteed to be available, ensuring the app never crashes due to API delays.*

-----

## 👥 Team Members (Group 8)

**Samsung Innovation Campus**

  * **Abdelbari Kecita** 
  * **Munieb Abdelrahman** 
  * **Hind Almutasim Hassan**
  * **Sharon Navaratnam**
  * **Raghad Sanosi**
  * **Wahed Shaik**
  

-----

## 📜 License & References

  * **Data Source:** [Open-Meteo API](https://open-meteo.com/) (Copernicus Sentinel-5P & ECMWF).
  * **Standards:** US EPA PM2.5 Air Quality Index.

<!-- end list -->

```
```
