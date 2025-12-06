import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import pickle


df = pd.read_csv('final_training_data_all_regions.csv')
df["time"] = pd.to_datetime(df["time"])
# regions: ['doha' 'khor' 'qatar' 'rayyan' 'wakrah' 'ummsalal']


def plot_timeseries(df, variable, regions="all"):
    """
    Plot a timeseries for any variable and any subset of regions.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns ['region', 'time', variable].
    variable : str
        The column to plot, e.g. "pm2_5 (μg/m³)".
    regions : str or list
        - "all" to plot every region
        - a single region name, e.g. "khor"
        - a list of regions, e.g. ["doha", "rayyan"]
    """
    if regions == "all":
        selected_regions = df['region'].unique()
    elif isinstance(regions, str):
        selected_regions = [regions]  # single region
    else:
        selected_regions = regions    # list of regions

    plt.figure(figsize=(14, 6))

    # Plot each region's timeseries
    for region in selected_regions:
        if region not in df['region'].unique():
            print(f"Warning: region '{region}' not found in dataset. Skipping.")
            continue

        region_data = df[df['region'] == region]
        plt.plot(region_data['time'], region_data[variable], label=region)

    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.title(f"{variable} Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_timeseries_resampled(df, variable, regions="all", freq="D"):
    """
    Plot a timeseries for any variable and any subset of regions with resampling.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns ['region', 'time', variable].
    variable : str
        The column to plot, e.g. "pm2_5 (μg/m³)".
    regions : str or list
        - "all" to plot every region
        - a single region name, e.g. "khor"
        - a list of regions, e.g. ["doha", "rayyan"]
    freq : str
        Resampling frequency (e.g. "D" for daily, "W" for weekly, "M" for monthly).
    """
    if regions == "all":
        selected_regions = df['region'].unique()
    elif isinstance(regions, str):
        selected_regions = [regions]  # single region
    else:
        selected_regions = regions    # list of regions

    plt.figure(figsize=(14, 6))

    # Plot each region's resampled timeseries
    for region in selected_regions:
        if region not in df['region'].unique():
            print(f"Warning: region '{region}' not found in dataset. Skipping.")
            continue

        region_data = df[df["region"] == region].set_index("time").sort_index()

        # Resample (default = daily mean)
        region_resampled = region_data[variable].resample(freq).mean()

        plt.plot(region_resampled.index, region_resampled.values, label=region)

    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.title(f"{variable} Over Time (Resampled: {freq})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# dataset has daily cycles: 24 hour seasonality
# also has weekly cycles: 7 x 24 = 168 hour seasonality
# might have long term trends as well


def fit_sarima(df, region, variable="pm2_5 (μg/m³)", seasonal_period=24, forecast_steps=None):
    """
    Automated SARIMA model fitting for a specific region and variable.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns ['region', 'time', variable].
    region : str
        The region to analyze.
    variable : str
        The column to model, default="pm2_5 (μg/m³)".
    seasonal_period : int
        The length of the seasonality, default=24 (hourly data with daily seasonality).
    forecast_steps : int or None
        If provided, generate and return forecast for this many steps.

    Returns
    -------
    dict
        Contains 'model_results', 'order', 'seasonal_order', and optionally 'forecast'.
    """
    region_data = df[df['region'] == region].set_index('time').sort_index()
    auto = auto_arima(
            region_data[variable],
            seasonal=True,
            m=seasonal_period,
            trace=True,
        )
    order = auto.order
    seasonal_order = auto.seasonal_order
    print(f"Selected order: {order}, seasonal_order: {seasonal_order}")

    print(f"Fitting SARIMA model for {region}...")
    model = SARIMAX(
        region_data[variable],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print(results.summary())

    output = {
        'model_results': results,
        'order': order,
        'seasonal_order': seasonal_order
    }
    
    if forecast_steps is not None:
        forecast = results.forecast(steps=forecast_steps)
        output['forecast'] = forecast
        print(f"\nForecast for next {forecast_steps} steps:")
        print(forecast)
    
    return output


def fit_sarima_all_regions(df, variable="pm2_5 (μg/m³)", seasonal_period=24):
    """
    Fit SARIMA models for all regions in the dataset.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns ['region', 'time', variable].
    variable : str
        The column to model.
    seasonal_period : int
        The length of the seasonality.

    Returns
    -------
    dict
        Dictionary mapping region names to their model results.
    """
    regions = df['region'].unique()
    models = {}
    
    for region in regions:
        print(f"\n{'='*60}")
        print(f"Processing region: {region}")
        print('='*60)
        try:
            models[region] = fit_sarima(
                df, region, variable, seasonal_period
            )
        except Exception as e:
            print(f"Error fitting model for {region}: {e}")
            models[region] = None
    
    return models


def save_models(models, filename='sarima_models.pkl'):
    """
    Save fitted SARIMA models to a file.

    Parameters
    ----------
    models : dict or single model result
        Dictionary of models (output from fit_sarima_all_regions) OR 
        a single model result (output from fit_sarima_auto).
    filename : str
        The file path to save the models.
    """
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {filename}")


def load_models(filename='sarima_models.pkl'):
    """
    Load fitted SARIMA models from a file.

    Parameters
    ----------
    filename : str
        The file path to load the models from.

    Returns
    -------
    dict or single model result
        Loaded model(s) in the same format they were saved.
    """
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    print(f"Models loaded from {filename}")
    return models


def plot_forecast(df, region, model_results, variable="pm2_5 (μg/m³)", 
                  forecast_steps=24, show_history_hours=168):
    """
    Plot historical data and forecast for a single region.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns ['region', 'time', variable].
    region : str
        The region to plot.
    model_results : SARIMAX results object
        The fitted model results.
    variable : str
        The column that was modeled.
    forecast_steps : int
        Number of steps to forecast into the future.
    show_history_hours : int
        How many hours of historical data to show (default=168, one week).
    """
    region_data = df[df['region'] == region].set_index('time').sort_index()
    
    # Get the last N hours of historical data
    history = region_data[variable].tail(show_history_hours)
    
    # Generate forecast
    forecast = model_results.forecast(steps=forecast_steps)
    
    # Create future time index
    last_time = history.index[-1]
    freq = pd.infer_freq(history.index)
    if freq is None:
        freq = 'H'  # Default to hourly
    future_index = pd.date_range(start=last_time, periods=forecast_steps+1, freq=freq)[1:]
    forecast.index = future_index
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(history.index, history.values, label='Historical', color='blue')
    plt.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
    
    # Add vertical line at forecast start
    plt.axvline(x=last_time, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.title(f"{variable} - Historical and Forecast for {region}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


result = fit_sarima(df, region="doha", forecast_steps=24)
save_models(result, "sarima_doha_model.pkl")

loaded_model = load_models(filename='sarima_doha_model.pkl')
plot_forecast(df, "doha", loaded_model['model_results'], forecast_steps=24)
