import streamlit as st
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pydeck as pdk
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Qatar Air Quality AI Forecaster", page_icon="🌍", layout="wide")

# Map Display Names to Model Feature Names
LOCATIONS = {
    "Ad-Dawhah (Doha)": {"lat": 25.2855, "lon": 51.531, "id": "city_doha"},
    "Al Khor": {"lat": 25.6839, "lon": 51.5058, "id": "city_khor"},
    "Al Rayyan": {"lat": 25.2919, "lon": 51.4244, "id": "city_rayyan"},
    "Al Wakrah": {"lat": 25.1715, "lon": 51.6034, "id": "city_wakrah"},
    "Umm Slal Ali": {"lat": 25.4697, "lon": 51.397498, "id": "city_ummsalal"},
    "Qatar (General)": {"lat": 25.5, "lon": 51.25, "id": "city_qatar"},
}

# EPA PM2.5 Thresholds
PM25_GOOD = 9.0
PM25_MODERATE = 35.4
PM25_UNHEALTHY_SENSITIVE = 55.4
PM25_UNHEALTHY = 125.4
PM25_VERY_UNHEALTHY = 225.4

MODEL_PATH = "universal_lstm_model.keras"

# Feature order must match training exactly
FEATURE_COLS = [
    'pm2_5 (μg/m³)', 'temperature_2m (°C)', 'relative_humidity_2m (%)',
    'wind_speed_10m (km/h)', 'wd_sin', 'wd_cos',
    'nitrogen_dioxide (μg/m³)', 'dust (μg/m³)', 'aerosol_optical_depth ()',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'city_doha', 'city_khor', 'city_qatar', 'city_rayyan', 'city_ummsalal', 'city_wakrah'
]

# ==========================================
# OPEN-METEO CLIENT SETUP
# ==========================================
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_pm25_color(pm25_value):
    """Return RGBA color list based on EPA PM2.5 thresholds."""
    if pm25_value is None or pd.isna(pm25_value):
        return [128, 128, 128, 140]
    if pm25_value <= PM25_GOOD:
        return [0, 228, 0, 140]
    elif pm25_value <= PM25_MODERATE:
        return [255, 255, 0, 140]
    elif pm25_value <= PM25_UNHEALTHY_SENSITIVE:
        return [255, 126, 0, 140]
    elif pm25_value <= PM25_UNHEALTHY:
        return [255, 0, 0, 140]
    elif pm25_value <= PM25_VERY_UNHEALTHY:
        return [143, 63, 151, 140]
    else:
        return [126, 0, 35, 140]

@st.cache_resource
def load_lstm_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data(ttl=3600)
def fetch_sdk_data(lat, lon):
    """
    Fetches data using openmeteo_requests SDK.
    SIMULATION LOGIC:
    - Reference Time ("Now") = Real Time - 2 Days
    - Start Date = Reference Time - 3 Days (To ensure 48h context)
    - End Date = Reference Time + 1 Day (To get "Future" weather for prediction)
    """
    
    # 1. Calculate Dates (UTC/GMT+0)
    real_now = datetime.now(timezone.utc)
    simulated_now = real_now - timedelta(days=2) # The "2 days ago" reference
    
    # We need:
    # - 48 hours BEFORE simulated_now for input
    # - 24 hours AFTER simulated_now for "future weather" features
    
    start_date = (simulated_now - timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (simulated_now + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # ---------------------------------------------------------
    # A. FETCH WEATHER (Archive API)
    # ---------------------------------------------------------
    url_weather = "https://archive-api.open-meteo.com/v1/archive"
    params_weather = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]
    }
    
    try:
        w_responses = openmeteo.weather_api(url_weather, params=params_weather)
        w_response = w_responses[0]
        
        w_hourly = w_response.Hourly()
        w_data = {"date": pd.date_range(
            start=pd.to_datetime(w_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(w_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=w_hourly.Interval()),
            inclusive="left"
        )}
        w_data["temperature_2m (°C)"] = w_hourly.Variables(0).ValuesAsNumpy()
        w_data["relative_humidity_2m (%)"] = w_hourly.Variables(1).ValuesAsNumpy()
        w_data["wind_speed_10m (km/h)"] = w_hourly.Variables(2).ValuesAsNumpy()
        w_data["wind_direction_10m (°)"] = w_hourly.Variables(3).ValuesAsNumpy()
        
        df_weather = pd.DataFrame(data=w_data)

        # ---------------------------------------------------------
        # B. FETCH AIR QUALITY
        # ---------------------------------------------------------
        url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params_aq = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["pm2_5", "nitrogen_dioxide", "dust", "aerosol_optical_depth"],
            "start_date": start_date,
            "end_date": end_date,
        }
        
        aq_responses = openmeteo.weather_api(url_aq, params=params_aq)
        aq_response = aq_responses[0]
        
        aq_hourly = aq_response.Hourly()
        aq_data = {"date": pd.date_range(
            start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=aq_hourly.Interval()),
            inclusive="left"
        )}
        
        # Order must match params list above
        aq_data["pm2_5 (μg/m³)"] = aq_hourly.Variables(0).ValuesAsNumpy()
        aq_data["nitrogen_dioxide (μg/m³)"] = aq_hourly.Variables(1).ValuesAsNumpy()
        aq_data["dust (μg/m³)"] = aq_hourly.Variables(2).ValuesAsNumpy()
        aq_data["aerosol_optical_depth ()"] = aq_hourly.Variables(3).ValuesAsNumpy()
        
        df_aq = pd.DataFrame(data=aq_data)
        
        # ---------------------------------------------------------
        # C. MERGE
        # ---------------------------------------------------------
        df = pd.merge(df_weather, df_aq, on="date", how="inner")
        df.rename(columns={"date": "datetime"}, inplace=True)
        
        return df, simulated_now

    except Exception as e:
        st.error(f"Open-Meteo SDK Error: {e}")
        return None, None

def prepare_features(df, city_key):
    df = df.copy()
    
    # 1. Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    
    # 2. Wind Features
    if 'wind_direction_10m (°)' in df.columns:
        wd_rad = df['wind_direction_10m (°)'] * np.pi / 180
        df['wd_sin'] = np.sin(wd_rad)
        df['wd_cos'] = np.cos(wd_rad)
    
    # 3. City One-Hot
    city_id = LOCATIONS[city_key]["id"]
    city_cols = ['city_doha', 'city_khor', 'city_qatar', 'city_rayyan', 'city_ummsalal', 'city_wakrah']
    for c in city_cols:
        df[c] = 1.0 if c == city_id else 0.0
        
    return df

def generate_forecast(model, df_history, df_future_weather, scaler_X, scaler_y):
    # Get last 48 hours strictly
    history_feats = df_history[FEATURE_COLS].tail(48)
    
    if len(history_feats) < 48:
        return pd.DataFrame() 
    
    current_seq_scaled = scaler_X.transform(history_feats)
    current_seq_scaled = current_seq_scaled.reshape(1, 48, 19) 
    
    predictions = []
    
    # Predict next 24 hours
    for i in range(24):
        # 1. Predict
        pred_scaled = model.predict(current_seq_scaled, verbose=0)[0][0]
        pred_val = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        
        # 2. Get Next Hour Weather (from our "Future" slice which is actually historic data)
        if i >= len(df_future_weather):
            break
        next_hour_weather = df_future_weather.iloc[i]
        
        # 3. Construct Row
        new_row = {}
        for col in FEATURE_COLS:
            if col == 'pm2_5 (μg/m³)':
                new_row[col] = pred_val
            elif col in next_hour_weather:
                new_row[col] = next_hour_weather[col]
            else:
                new_row[col] = df_history.iloc[-1][col]
        
        new_row_df = pd.DataFrame([new_row], columns=FEATURE_COLS)
        new_row_scaled = scaler_X.transform(new_row_df)
        
        # 4. Update Sequence
        current_seq_scaled = np.concatenate([current_seq_scaled[:, 1:, :], new_row_scaled.reshape(1, 1, 19)], axis=1)
        
        pred_time = df_history['datetime'].iloc[-1] + timedelta(hours=i+1)
        predictions.append({
            "datetime": pred_time,
            "Predicted PM2.5": max(0, pred_val),
            "Type": "Forecast"
        })
        
    return pd.DataFrame(predictions)

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.title("🌍 Qatar Air Quality AI Forecaster")
    
    st.sidebar.header("Settings")
    selected_city_name = st.sidebar.selectbox("Select Region", list(LOCATIONS.keys()))
    selected_loc_data = LOCATIONS[selected_city_name]
    
    model = load_lstm_model()

    # Data Fetching
    with st.spinner("Fetching Data (Simulation Mode: 2 Days Lag)..."):
        raw_df, simulated_now_utc = fetch_sdk_data(selected_loc_data['lat'], selected_loc_data['lon'])
    
    if raw_df is not None and not raw_df.empty:
        
        # Determine Cutoff (Simulated Now)
        # Ensure simulated_now_utc has no timezone info for pandas comparison if df is tz-naive, 
        # but fetch_sdk_data returns UTC aware. Let's ensure alignment.
        cutoff_dt = pd.to_datetime(simulated_now_utc)
        
        st.info(f"ℹ️ **Simulation Mode Active**: System is running as if the date is **{cutoff_dt.strftime('%Y-%m-%d %H:00')} UTC**. (Using 2-day old data to predict 'Yesterday')")

        # Feature Engineering
        processed_df = prepare_features(raw_df, selected_city_name)
        
        # Split Data
        # History: Everything UP TO the simulated "Now"
        history_df = processed_df[processed_df['datetime'] <= cutoff_dt].dropna(subset=['pm2_5 (μg/m³)'])
        
        # Future Weather: The data AFTER the simulated "Now" (which we actually have because it's in the past)
        future_weather_df = processed_df[processed_df['datetime'] > cutoff_dt]

        forecast_df = pd.DataFrame()
        
        if len(history_df) >= 48:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            # Fit scaler on history
            scaler_X.fit(history_df[FEATURE_COLS])
            scaler_y.fit(history_df[['pm2_5 (μg/m³)']])
            
            if model:
                forecast_df = generate_forecast(model, history_df, future_weather_df, scaler_X, scaler_y)
        
        # --- DISPLAY METRICS ---
        if not history_df.empty:
            latest = history_df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current PM2.5", f"{latest['pm2_5 (μg/m³)']:.1f}", delta_color="inverse")
            c2.metric("Temp", f"{latest['temperature_2m (°C)']:.1f} °C")
            c3.metric("Humidity", f"{latest['relative_humidity_2m (%)']:.0f}%")
            c4.metric("Wind", f"{latest['wind_speed_10m (km/h)']:.1f} km/h")

        st.markdown("---")
        
        # --- MAP & SLIDER ---
        st.subheader("🗺️ PM2.5 Forecast Map (Next 24 Hours)")
        hour_offset = st.slider(f"Forecast Hour (0 = {cutoff_dt.strftime('%H:00')})", 0, 24, 0)
        
        display_val = 0
        display_time = "N/A"
        label_type = "NO DATA"
        
        if hour_offset == 0:
            if not history_df.empty:
                display_val = history_df.iloc[-1]['pm2_5 (μg/m³)']
                display_time = history_df.iloc[-1]['datetime'].strftime('%Y-%m-%d %H:00 UTC')
                label_type = "ACTUAL (Historic)"
        else:
            if not forecast_df.empty and hour_offset <= len(forecast_df):
                row = forecast_df.iloc[hour_offset - 1]
                display_val = row['Predicted PM2.5']
                display_time = row['datetime'].strftime('%Y-%m-%d %H:00 UTC')
                label_type = "AI FORECAST"

        color = get_pm25_color(display_val)
        
        # PyDeck Map
        view_state = pdk.ViewState(latitude=selected_loc_data["lat"], longitude=selected_loc_data["lon"], zoom=10)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lat": [selected_loc_data["lat"]], "lon": [selected_loc_data["lon"]], "color": [color]}),
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=3000,
            pickable=True,
            filled=True
        )
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        with c2:
            st.info(f"**{label_type}**\n\nTime: {display_time}\n\nPM2.5: **{display_val:.2f}**")
            
        # --- CHART ---
        st.subheader("📈 24-Hour Forecast Trend")
        if not forecast_df.empty:
            chart_hist = history_df.tail(24)[['datetime', 'pm2_5 (μg/m³)']].rename(columns={'pm2_5 (μg/m³)': 'Value'})
            chart_hist['Type'] = 'Actual'
            chart_fore = forecast_df.rename(columns={'Predicted PM2.5': 'Value'})[['datetime', 'Value', 'Type']]
            st.line_chart(pd.concat([chart_hist, chart_fore]), x="datetime", y="Value", color="Type")
            
        with st.expander("View Raw Data Table"):
            st.dataframe(raw_df)

    else:
        st.error("Failed to load data.")

if __name__ == "__main__":
    main()