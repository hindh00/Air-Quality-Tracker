# Qatar Air Quality AI Forecaster

A deep learning system that predicts PM2.5 concentrations (fine particulate matter) with 24-hour lead time across six municipalities in Qatar. The system integrates satellite data, meteorological information, and ground-level simulations to provide actionable air quality forecasts.

# Project Features
- 24-Hour PM2.5 Forecasting: Accurate predictions with 92% global accuracy

- Multi-Region Coverage: Doha, Al Khor, Al Rayyan, Al Wakrah, Umm Slal Ali, and Qatar (General)

- Real-time Dashboard: Interactive Streamlit web application with 3D mapping

- AI-Powered Insights: Bidirectional LSTM model trained on historical environmental data

- EPA Standards: Color-coded alerts based on US Environmental Protection Agency thresholds

- Simulation Mode: Test forecasting capabilities using historical data


# Key Data Features (19 Total)
- Target: PM2.5 concentration
- Meteorological: Temperature, Humidity, Wind Speed/Direction
- Pollution Indicators: NO₂, Dust, Aerosol Optical Depth
- Cyclical Time: Hour (sin/cos), Month (sin/cos)
- Geospatial: One-hot encoded city locations

# Quick Start

## Prerequisites
- Python 3.9+
- TensorFlow 2.13+
- Streamlit 1.28+

## Installation
- Install dependencies:
  * pip install -r requirements.txt
  
- Run the dashboard:
  * streamlit run Dashboard/app.py

# Usage
## Web Dashboard
- Select a region from the sidebar dropdown
- View current air quality metrics (simulated with 2-day lag)
- Use the slider to explore 24-hour forecasts
- Observe the 3D map showing PM2.5 concentrations
- Analyze the forecast trend chart

# Model Development
## Training Process
- Data Collection: Historical data from Open-Meteo API (2023-2025)

- Feature Engineering: 19 features including cyclical time encoding

- Model Architecture: Bidirectional LSTM with attention mechanism

- Training: 100 epochs with early stopping and dropout regularization

- Validation: Hold-out test set with temporal cross-validation

# API Integration
The system uses Open-Meteo for:

- Weather Data: Temperature, humidity, wind speed/direction

- Air Quality: PM2.5, NO₂, dust, aerosol optical depth

- Historical Data: ECMWF reanalysis and Copernicus satellite data

🏆 Results
