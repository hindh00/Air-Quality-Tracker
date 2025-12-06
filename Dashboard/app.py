import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# Configuration
LOCATIONS = {
    "Ad-Dawhah (Doha)": {"lat": 25.2855, "lon": 51.531},
    "Al Khor": {"lat": 25.6839, "lon": 51.5058},
    "Al Rayyan": {"lat": 25.2919, "lon": 51.4244},
    "Al Wakrah": {"lat": 25.1715, "lon": 51.6034},
    "Umm Slal Ali": {"lat": 25.4697, "lon": 51.397498},
    "Qatar (General)": {"lat": 25.5, "lon": 51.25},
}

# EPA PM2.5 Thresholds (May 2024 Standards)
PM25_GOOD = 9.0
PM25_MODERATE = 35.4
PM25_UNHEALTHY_SENSITIVE = 55.4
PM25_UNHEALTHY = 125.4
PM25_VERY_UNHEALTHY = 225.4
# Anything above 225.4 is HAZARDOUS


def get_pm25_color(pm25_value):
    """Return RGBA color list based on EPA PM2.5 thresholds."""
    if pm25_value is None:
        return [128, 128, 128, 200]  # Grey (No Data)
    if pm25_value <= PM25_GOOD:
        return [0, 228, 0, 200]  # Green (Good)
    elif pm25_value <= PM25_MODERATE:
        return [255, 255, 0, 200]  # Yellow (Moderate)
    elif pm25_value <= PM25_UNHEALTHY_SENSITIVE:
        return [255, 126, 0, 200]  # Orange (Unhealthy for Sensitive)
    elif pm25_value <= PM25_UNHEALTHY:
        return [255, 0, 0, 200]  # Red (Unhealthy)
    elif pm25_value <= PM25_VERY_UNHEALTHY:
        return [143, 63, 151, 200]  # Purple (Very Unhealthy)
    else:
        return [126, 0, 35, 200]  # Maroon (Hazardous)


def get_wind_direction(degrees):
    """Convert degrees to cardinal direction."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(degrees / 45) % 8
    return directions[idx]


@st.cache_data(ttl=3600)
def fetch_weather_data(lat, lon, start_date, end_date):
    """Fetch weather data from Open-Meteo API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": "Asia/Qatar",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API Error: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_air_quality_data(lat, lon, start_date, end_date):
    """Fetch air quality data from Open-Meteo API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust",
        "timezone": "Asia/Qatar",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Air Quality API Error: {e}")
        return None


def process_data(weather_data, air_quality_data):
    """Process and combine weather and air quality data into a DataFrame."""
    if not weather_data or not air_quality_data:
        return None

    # Create weather DataFrame
    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather_data["hourly"]["time"]),
        "temperature_2m": weather_data["hourly"]["temperature_2m"],
        "relative_humidity_2m": weather_data["hourly"]["relative_humidity_2m"],
        "wind_speed_10m": weather_data["hourly"]["wind_speed_10m"],
        "wind_direction_10m": weather_data["hourly"]["wind_direction_10m"],
        "precipitation": weather_data["hourly"]["precipitation"],
    })

    # Create air quality DataFrame
    aq_df = pd.DataFrame({
        "datetime": pd.to_datetime(air_quality_data["hourly"]["time"]),
        "pm10": air_quality_data["hourly"]["pm10"],
        "pm2_5": air_quality_data["hourly"]["pm2_5"],
        "carbon_monoxide": air_quality_data["hourly"]["carbon_monoxide"],
        "nitrogen_dioxide": air_quality_data["hourly"]["nitrogen_dioxide"],
        "sulphur_dioxide": air_quality_data["hourly"]["sulphur_dioxide"],
        "ozone": air_quality_data["hourly"]["ozone"],
        "aerosol_optical_depth": air_quality_data["hourly"]["aerosol_optical_depth"],
        "dust": air_quality_data["hourly"]["dust"],
    })

    # Merge on datetime
    combined_df = pd.merge(weather_df, aq_df, on="datetime", how="outer")
    combined_df = combined_df.sort_values("datetime", ascending=False)
    
    return combined_df


def get_latest_readings(df):
    """Get the most recent non-null readings."""
    if df is None or df.empty:
        return None
    
    latest = {}
    for col in df.columns:
        if col != "datetime":
            non_null = df[df[col].notna()][col]
            if not non_null.empty:
                latest[col] = non_null.iloc[0]
            else:
                latest[col] = None
    
    # Get the most recent datetime
    latest["datetime"] = df["datetime"].iloc[0]
    return latest


def main():
    st.set_page_config(
        page_title="Qatar Air Quality Monitor",
        page_icon="🌍",
        layout="wide"
    )

    st.title("🌍 Qatar Air Quality Monitor")

    st.sidebar.header("Settings")
    selected_city = st.sidebar.selectbox(
        "Select City",
        options=list(LOCATIONS.keys()),
        index=0
    )

    lat = LOCATIONS[selected_city]["lat"]
    lon = LOCATIONS[selected_city]["lon"]

    #! Date range (past 7 days | ending yesterday for weather data availability)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")

    st.sidebar.info(f"📍 Coordinates: {lat}, {lon}")
    st.sidebar.info(f"🗓️ Data: {start_date} to {end_date}")

    # Fetch data
    with st.spinner("Fetching data from Open-Meteo..."):
        weather_data = fetch_weather_data(lat, lon, start_date, end_date)
        air_quality_data = fetch_air_quality_data(lat, lon, start_date, end_date)

    if weather_data and air_quality_data:
        # Process data
        df = process_data(weather_data, air_quality_data)
        latest = get_latest_readings(df)

        if df is not None and latest:
            pm25_value = latest.get("pm2_5")
            marker_color = get_pm25_color(pm25_value)

            # Display current PM2.5 status
            col1, col2, col3 = st.columns(3)
            with col1:
                pm25_display = f"{pm25_value:.1f}" if pm25_value else "N/A"
                st.metric("PM2.5 (μg/m³)", pm25_display)
            with col2:
                temp = latest.get("temperature_2m")
                temp_display = f"{temp:.1f}°C" if temp else "N/A"
                st.metric("Temperature", temp_display)
            with col3:
                wind = latest.get("wind_speed_10m")
                wind_display = f"{wind:.1f} km/h" if wind else "N/A"
                st.metric("Wind Speed", wind_display)

            st.subheader(f"📍 {selected_city} Location")
            
            map_data = pd.DataFrame({
                "lat": [lat],
                "lon": [lon],
                "color": [marker_color],
                "size": [2000]
            })

            st.map(map_data, latitude="lat", longitude="lon", size="size", color="color")

            # Trend Chart
            st.subheader("📉 PM2.5 Trend (Last 7 Days)")
            chart_df = df.sort_values("datetime").set_index("datetime")[["pm2_5", "pm10"]]
            st.area_chart(chart_df, color=["#FF4B4B", "#1f77b4"])

            if pm25_value:
                if pm25_value <= PM25_GOOD:
                    st.success(f"✅ Air Quality: GOOD (PM2.5: {pm25_value:.1f} μg/m³)")
                elif pm25_value <= PM25_MODERATE:
                    st.warning(f"⚠️ Air Quality: MODERATE (PM2.5: {pm25_value:.1f} μg/m³)")
                elif pm25_value <= PM25_UNHEALTHY_SENSITIVE:
                    st.warning(f"🤧 Air Quality: UNHEALTHY FOR SENSITIVE GROUPS (PM2.5: {pm25_value:.1f} μg/m³)")
                elif pm25_value <= PM25_UNHEALTHY:
                    st.error(f"🚫 Air Quaality: UNHEALTHY (PM2.5: {pm25_value:.1f} μg/m³)")
                elif pm25_value <= PM25_VERY_UNHEALTHY:
                    st.error(f"😷 Air Quality: VERY UNHEALTHY (PM2.5: {pm25_value:.1f} μg/m³)")
                else:
                    st.error(f"🚨 Air Quality: HAZARDOUS (PM2.5: {pm25_value:.1f} μg/m³)")

            # Data Table
            st.subheader("📊 Latest Readings")
            
            # Format latest readings for display
            readings_display = {
                "Parameter": [
                    "Temperature (°C)",
                    "Relative Humidity (%)",
                    "Wind Speed (km/h)",
                    "Wind Direction (°)",
                    "Precipitation (mm)",
                    "PM10 (μg/m³)",
                    "PM2.5 (μg/m³)",
                    "Carbon Monoxide (μg/m³)",
                    "Nitrogen Dioxide (μg/m³)",
                    "Sulphur Dioxide (μg/m³)",
                    "Ozone (μg/m³)",
                    "Aerosol Optical Depth",
                    "Dust (μg/m³)"
                ],
                "Value": [
                    f"{latest.get('temperature_2m', 'N/A'):.1f}" if latest.get('temperature_2m') else "N/A",
                    f"{latest.get('relative_humidity_2m', 'N/A'):.0f}" if latest.get('relative_humidity_2m') else "N/A",
                    f"{latest.get('wind_speed_10m', 'N/A'):.1f}" if latest.get('wind_speed_10m') else "N/A",
                    f"{get_wind_direction(latest.get('wind_direction_10m'))} ({latest.get('wind_direction_10m'):.0f}°)" if latest.get('wind_direction_10m') else "N/A",
                    f"{latest.get('precipitation', 'N/A'):.2f}" if latest.get('precipitation') is not None else "N/A",
                    f"{latest.get('pm10', 'N/A'):.1f}" if latest.get('pm10') else "N/A",
                    f"{latest.get('pm2_5', 'N/A'):.1f}" if latest.get('pm2_5') else "N/A",
                    f"{latest.get('carbon_monoxide', 'N/A'):.1f}" if latest.get('carbon_monoxide') else "N/A",
                    f"{latest.get('nitrogen_dioxide', 'N/A'):.1f}" if latest.get('nitrogen_dioxide') else "N/A",
                    f"{latest.get('sulphur_dioxide', 'N/A'):.1f}" if latest.get('sulphur_dioxide') else "N/A",
                    f"{latest.get('ozone', 'N/A'):.1f}" if latest.get('ozone') else "N/A",
                    f"{latest.get('aerosol_optical_depth', 'N/A'):.3f}" if latest.get('aerosol_optical_depth') else "N/A",
                    f"{latest.get('dust', 'N/A'):.1f}" if latest.get('dust') else "N/A",
                ]
            }
            
            readings_df = pd.DataFrame(readings_display)
            st.dataframe(readings_df, width="stretch", hide_index=True)

            # Section for full hourly data
            with st.expander("📈 View Full Hourly Data (Last 7 Days)"):
                display_df = df.copy()
                display_df.columns = [
                    "Datetime", "Temp (°C)", "Humidity (%)", "Wind (km/h)", 
                    "Wind Dir (°)", "Precip (mm)", "PM10", "PM2.5", 
                    "CO", "NO2", "SO2", "O3", "AOD", "Dust"
                ]
                st.dataframe(display_df, width="stretch", hide_index=True)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{selected_city}_air_quality.csv",
                    mime="text/csv",
                )

    else:
        st.error("Failed to fetch data. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
