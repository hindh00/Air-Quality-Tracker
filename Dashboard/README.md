# Qatar Air Quality Monitor Dashboard

A Streamlit dashboard that displays real-time air quality and weather data for cities in Qatar using the Open-Meteo API.

## Features

-   **City Selection**: Monitor 6 locations across Qatar (Doha, Al Khor, Al Rayyan, Al Wakrah, Umm Slal Ali, Qatar General)
-   **Interactive Map**: Color-coded marker based on PM2.5 levels (EPA 2024 standards)
-   **PM2.5 Trend Chart**: 7-day historical visualization
-   **Live Metrics**: Temperature, wind speed, humidity, and all air quality parameters
-   **Data Export**: Download CSV for further analysis

## Air Quality Categories (EPA PM2.5 Standards)

| Level                   | PM2.5 (μg/m³) | Color     |
| ----------------------- | ------------- | --------- |
| Good                    | 0 - 9.0       | 🟢 Green  |
| Moderate                | 9.1 - 35.4    | 🟡 Yellow |
| Unhealthy for Sensitive | 35.5 - 55.4   | 🟠 Orange |
| Unhealthy               | 55.5 - 125.4  | 🔴 Red    |
| Very Unhealthy          | 125.5 - 225.4 | 🟣 Purple |
| Hazardous               | > 225.4       | 🟤 Maroon |

## Installation

### Option 1: Using requirements.txt (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Direct install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Data Sources

-   **Weather**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
-   **Air Quality**: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)

## Parameters Tracked

**Weather**: Temperature, Humidity, Wind Speed/Direction, Precipitation

**Air Quality**: PM2.5, PM10, CO, NO2, SO2, O3, Aerosol Optical Depth, Dust
