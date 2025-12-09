import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')

# ==================== CREATE MODEL DIRECTORY ====================
MODEL_DIR = "LSTM_model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")
else:
    print(f"Using existing directory: {MODEL_DIR}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== DATA LOADING ====================
print("Loading data...")
df = pd.read_csv('final_training_data_all_regions.csv', parse_dates=['time'])
df = df.sort_values('time')  # Ensure chronological order

# Filter for Doha region only (if you want to focus on one region)
if 'region' in df.columns:
    df = df[df['region'] == 'doha'].copy()
    print(f"Filtered to Doha region: {len(df)} samples")

print(f"Data Shape: {df.shape}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Total hours: {len(df)}")

# ==================== DATA PREPARATION ====================
print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

# Select features for LSTM (focus on most relevant ones)
selected_features = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'wind_speed_10m (km/h)',
    'wind_direction_10m (°)',
    'pm10 (μg/m³)',
    'carbon_monoxide (μg/m³)',
    'nitrogen_dioxide (μg/m³)',
    'ozone (μg/m³)',
    'dust (μg/m³)',
    'pm2_5 (μg/m³)'  # Target will be separated later
]

# Keep only selected features and time
df_lstm = df[['time'] + selected_features].copy()

# Create time-based features
df_lstm['hour'] = df_lstm['time'].dt.hour
df_lstm['day_of_week'] = df_lstm['time'].dt.dayofweek
df_lstm['month'] = df_lstm['time'].dt.month

# Cyclical encoding
df_lstm['hour_sin'] = np.sin(2 * np.pi * df_lstm['hour'] / 24)
df_lstm['hour_cos'] = np.cos(2 * np.pi * df_lstm['hour'] / 24)
df_lstm['day_sin'] = np.sin(2 * np.pi * df_lstm['day_of_week'] / 7)
df_lstm['day_cos'] = np.cos(2 * np.pi * df_lstm['day_of_week'] / 7)
df_lstm['month_sin'] = np.sin(2 * np.pi * df_lstm['month'] / 12)
df_lstm['month_cos'] = np.cos(2 * np.pi * df_lstm['month'] / 12)

# Add cyclical features to selected features list
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

print(f"\nNumber of original features: {len(selected_features)}")
print(f"Number of time features: {len(time_features)}")

# Drop rows with missing values
df_lstm_clean = df_lstm.dropna().copy()
print(f"\nData after cleaning: {df_lstm_clean.shape} (removed {len(df_lstm) - len(df_lstm_clean)} rows)")

# ==================== CREATE SEQUENCES (CORRECTED) ====================
def create_sequences_numpy(data_array, sequence_length=24, forecast_horizon=1):
    """
    Create sequences from numpy array for LSTM training
    
    Args:
        data_array: numpy array with all columns (features + target)
        sequence_length: Number of past hours to use
        forecast_horizon: Number of hours ahead to predict
    
    Returns:
        X: Sequences of features (samples, timesteps, features)
        y: Target values
    """
    X, y = [], []
    
    for i in range(len(data_array) - sequence_length - forecast_horizon + 1):
        # Get sequence of features (all columns except last)
        sequence = data_array[i:i + sequence_length, :-1]  # All columns except last
        # Get target (last column) at forecast horizon
        target = data_array[i + sequence_length + forecast_horizon - 1, -1]
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)

# Prepare data for sequences
feature_cols = [col for col in selected_features if col != 'pm2_5 (μg/m³)']
feature_cols.extend(time_features)  # Add time features

print(f"\nTotal features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Create feature matrix and target
X_data = df_lstm_clean[feature_cols].values
y_data = df_lstm_clean['pm2_5 (μg/m³)'].values.reshape(-1, 1)

print(f"\nX_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")

# Scale features and target separately
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Scale features
X_scaled = scaler_features.fit_transform(X_data)

# Scale target
y_scaled = scaler_target.fit_transform(y_data)

# Combine scaled data (for sequence creation)
scaled_data = np.column_stack([X_scaled, y_scaled])

# Create sequences
SEQUENCE_LENGTH = 24  # Use 24 hours of history
FORECAST_HORIZON = 1  # Predict 1 hour ahead

X_sequences, y_sequences = create_sequences_numpy(
    scaled_data, 
    sequence_length=SEQUENCE_LENGTH, 
    forecast_horizon=FORECAST_HORIZON
)

print(f"\nSequence shape: {X_sequences.shape}")
print(f"Target shape: {y_sequences.shape}")
print(f"Number of sequences: {len(X_sequences)}")

# ==================== TRAIN-TEST SPLIT ====================
# Temporal split for time series
split_idx = int(len(X_sequences) * 0.8)
X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

# Further split for validation
val_split = int(len(X_train) * 0.8)
X_train_final, X_val = X_train[:val_split], X_train[val_split:]
y_train_final, y_val = y_train[:val_split], y_train[val_split:]

print(f"\nTraining set: {X_train_final.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# ==================== LSTM MODEL ARCHITECTURE ====================
print("\n" + "="*50)
print("BUILDING LSTM MODEL")
print("="*50)

# Model parameters
INPUT_SHAPE = (SEQUENCE_LENGTH, len(feature_cols))
N_FEATURES = len(feature_cols)

# Clear any existing TensorFlow session
keras.backend.clear_session()

# Build LSTM model
model = keras.Sequential([
    layers.LSTM(64, 
                activation='tanh',
                return_sequences=True,
                input_shape=INPUT_SHAPE,
                dropout=0.2,
                recurrent_dropout=0.2),
    
    layers.LSTM(32,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2),
    
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(1)
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

print("\nModel Summary:")
model.summary()

# ==================== CALLBACKS ====================
# Define callbacks for training
model_checkpoint_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
callbacks_list = [
    # Early stopping to prevent overfitting
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when plateau
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    ),
    
    # Model checkpoint
    callbacks.ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ==================== TRAIN MODEL ====================
print("\n" + "="*50)
print("TRAINING LSTM MODEL")
print("="*50)

# Train the model
history = model.fit(
    X_train_final, y_train_final,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks_list,
    verbose=1
)

# ==================== EVALUATION ====================
# Load best model
try:
    model = keras.models.load_model(model_checkpoint_path)
    print("Loaded best model from checkpoint")
except:
    print("Using final trained model")

# Make predictions
print("\nMaking predictions...")
y_pred_train = model.predict(X_train_final, verbose=0).flatten()
y_pred_val = model.predict(X_val, verbose=0).flatten()
y_pred_test = model.predict(X_test, verbose=0).flatten()

# Inverse transform predictions and actual values
y_train_actual = scaler_target.inverse_transform(y_train_final.reshape(-1, 1)).flatten()
y_train_pred = scaler_target.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()

y_val_actual = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_val_pred = scaler_target.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred = scaler_target.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Calculate metrics
def calculate_lstm_metrics(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MAE: {mae:.2f} μg/m³")
    print(f"  RMSE: {rmse:.2f} μg/m³")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

train_metrics = calculate_lstm_metrics(y_train_actual, y_train_pred, "Training")
val_metrics = calculate_lstm_metrics(y_val_actual, y_val_pred, "Validation")
test_metrics = calculate_lstm_metrics(y_test_actual, y_test_pred, "Test")

# ==================== VISUALIZATION ====================
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 15))

# Plot 1: Training History
ax1 = plt.subplot(3, 3, 1)
if 'loss' in history.history:
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Plot 2: Test Set Predictions vs Actual (Scatter)
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_test_actual, y_test_pred, alpha=0.5, s=20, c='blue')
ax2.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', linewidth=2)
ax2.set_xlabel('Actual PM2.5 (μg/m³)')
ax2.set_ylabel('Predicted PM2.5 (μg/m³)')
ax2.set_title(f'Test Set: Predictions vs Actual (R²={test_metrics["r2"]:.3f})')
ax2.grid(True, alpha=0.3)

# Plot 3: Time Series Predictions (first 100 hours)
ax3 = plt.subplot(3, 3, 3)
hours_to_plot = min(100, len(y_test_actual))
ax3.plot(range(hours_to_plot), y_test_actual[:hours_to_plot], 
         'b-', label='Actual', linewidth=2, alpha=0.8)
ax3.plot(range(hours_to_plot), y_test_pred[:hours_to_plot], 
         'r-', label='Predicted', linewidth=2, alpha=0.8)
ax3.set_xlabel('Hour')
ax3.set_ylabel('PM2.5 (μg/m³)')
ax3.set_title(f'First {hours_to_plot} Hours of Test Set')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax4 = plt.subplot(3, 3, 4)
errors = y_test_actual - y_test_pred
ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Prediction Error (μg/m³)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Error Distribution\nMean: {errors.mean():.2f}, Std: {errors.std():.2f}')
ax4.grid(True, alpha=0.3)

# Plot 5: Residual Plot
ax5 = plt.subplot(3, 3, 5)
residuals = y_test_actual - y_test_pred
ax5.scatter(y_test_pred, residuals, alpha=0.5, s=20, c='green')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted PM2.5 (μg/m³)')
ax5.set_ylabel('Residuals')
ax5.set_title('Residual Plot')
ax5.grid(True, alpha=0.3)

# Plot 6: Learning Rate History
ax6 = plt.subplot(3, 3, 6)
if 'lr' in history.history:
    ax6.plot(history.history['lr'], linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate')
    ax6.set_title('Learning Rate Schedule')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

# Plot 7: MAE during training
ax7 = plt.subplot(3, 3, 7)
if 'mae' in history.history:
    ax7.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax7.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('MAE')
    ax7.set_title('MAE during Training')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

# Plot 8: Actual vs Predicted Distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(y_test_actual, bins=30, alpha=0.5, label='Actual', edgecolor='black', density=True)
ax8.hist(y_test_pred, bins=30, alpha=0.5, label='Predicted', edgecolor='black', density=True)
ax8.set_xlabel('PM2.5 (μg/m³)')
ax8.set_ylabel('Density')
ax8.set_title('Distribution: Actual vs Predicted')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Rolling Error
ax9 = plt.subplot(3, 3, 9)
if len(errors) > 24:
    rolling_error = pd.Series(errors).rolling(window=24).mean()
    ax9.plot(rolling_error, linewidth=2, color='purple')
    ax9.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax9.set_xlabel('Hour')
    ax9.set_ylabel('Rolling MAE (24h window)')
    ax9.set_title('Rolling Prediction Error')
    ax9.grid(True, alpha=0.3)
else:
    ax9.text(0.5, 0.5, 'Insufficient data\nfor rolling error',
             horizontalalignment='center', verticalalignment='center',
             transform=ax9.transAxes, fontsize=12)
    ax9.set_title('Rolling Prediction Error')

plt.tight_layout()
results_plot_path = os.path.join(MODEL_DIR, 'lstm_results.png')
plt.savefig(results_plot_path, dpi=100, bbox_inches='tight')
plt.show()

# ==================== FORECAST FUNCTION ====================
def forecast_pm25_lstm(model, scaler_features, scaler_target, recent_data, 
                       feature_cols, sequence_length=SEQUENCE_LENGTH):
    """
    Forecast PM2.5 using LSTM model
    
    Args:
        model: Trained LSTM model
        scaler_features: Fitted scaler for features
        scaler_target: Fitted scaler for target
        recent_data: Array of recent data (shape: sequence_length x n_features)
        feature_cols: List of feature column names
        sequence_length: Length of input sequence
    
    Returns:
        Predicted PM2.5 value (unscaled)
    """
    # Ensure recent_data has correct shape
    if recent_data.shape[0] < sequence_length:
        raise ValueError(f"Need at least {sequence_length} hours of data")
    
    # Use only the most recent sequence_length hours
    recent_data = recent_data[-sequence_length:]
    
    # Scale the features
    scaled_features = scaler_features.transform(recent_data)
    
    # Reshape for LSTM (1 sample, sequence_length timesteps, n_features)
    input_seq = scaled_features.reshape(1, sequence_length, len(feature_cols))
    
    # Make prediction
    scaled_prediction = model.predict(input_seq, verbose=0)[0][0]
    
    # Inverse transform the prediction
    prediction = scaler_target.inverse_transform([[scaled_prediction]])[0][0]
    
    return prediction

# ==================== MULTI-STEP FORECASTING ====================
def multi_step_forecast_lstm(model, initial_sequence, steps_ahead=168, 
                            scaler_features=scaler_features, 
                            scaler_target=scaler_target,
                            feature_cols=feature_cols,
                            df_history=None):
    """
    Make multi-step forecasts using iterative prediction
    
    Args:
        model: Trained LSTM model
        initial_sequence: Initial sequence of features
        steps_ahead: Number of steps to forecast
        scaler_features, scaler_target: Scalers
        feature_cols: Feature columns
        df_history: Historical dataframe for feature updates
    
    Returns:
        Array of forecasts, forecast dates
    """
    forecasts = []
    forecast_dates = []
    
    current_seq = initial_sequence.copy()
    last_date = df_lstm_clean['time'].iloc[-1]
    
    for step in range(steps_ahead):
        forecast_date = last_date + timedelta(hours=step+1)
        forecast_dates.append(forecast_date)
        
        # Get prediction for next step
        input_seq = current_seq.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        scaled_pred = model.predict(input_seq, verbose=0)[0][0]
        pred_value = scaler_target.inverse_transform([[scaled_pred]])[0][0]
        forecasts.append(pred_value)
        
        # For multi-step forecasting, we need to create new input
        # This is a simplified approach - in reality, you need future feature values
        
        # Create a new feature row for the next time step
        # Use the last row's features as a starting point
        if df_history is not None and step < len(df_history) - SEQUENCE_LENGTH:
            # Get actual future features if available (for testing)
            new_features = df_history[feature_cols].iloc[-SEQUENCE_LENGTH + step + 1].values
        else:
            # Create synthetic features (simplified)
            # In practice, you would need weather forecasts here
            new_features = current_seq[-1].copy()  # Use last known features
            
            # Update time features
            hour = forecast_date.hour
            day_of_week = forecast_date.dayofweek
            month = forecast_date.month
            
            # Find time feature indices
            time_feature_indices = {}
            for idx, col in enumerate(feature_cols):
                if 'hour_sin' in col:
                    time_feature_indices['hour_sin'] = idx
                elif 'hour_cos' in col:
                    time_feature_indices['hour_cos'] = idx
                elif 'day_sin' in col:
                    time_feature_indices['day_sin'] = idx
                elif 'day_cos' in col:
                    time_feature_indices['day_cos'] = idx
                elif 'month_sin' in col:
                    time_feature_indices['month_sin'] = idx
                elif 'month_cos' in col:
                    time_feature_indices['month_cos'] = idx
            
            # Update cyclical features
            if 'hour_sin' in time_feature_indices:
                new_features[time_feature_indices['hour_sin']] = np.sin(2 * np.pi * hour / 24)
            if 'hour_cos' in time_feature_indices:
                new_features[time_feature_indices['hour_cos']] = np.cos(2 * np.pi * hour / 24)
            if 'day_sin' in time_feature_indices:
                new_features[time_feature_indices['day_sin']] = np.sin(2 * np.pi * day_of_week / 7)
            if 'day_cos' in time_feature_indices:
                new_features[time_feature_indices['day_cos']] = np.cos(2 * np.pi * day_of_week / 7)
            if 'month_sin' in time_feature_indices:
                new_features[time_feature_indices['month_sin']] = np.sin(2 * np.pi * month / 12)
            if 'month_cos' in time_feature_indices:
                new_features[time_feature_indices['month_cos']] = np.cos(2 * np.pi * month / 12)
        
        # Update sequence: remove oldest, add new features
        current_seq = np.vstack([current_seq[1:], new_features.reshape(1, -1)])
    
    return np.array(forecasts), np.array(forecast_dates)

# ==================== GENERATE WEEKLY FORECAST ====================
print("\n" + "="*50)
print("GENERATING WEEKLY FORECAST")
print("="*50)

# Get the last SEQUENCE_LENGTH hours of data
last_sequence = X_scaled[-SEQUENCE_LENGTH:]

# Generate 1-week forecast
forecast_horizon = 168  # 1 week = 168 hours
try:
    lstm_forecasts, lstm_forecast_dates = multi_step_forecast_lstm(
        model=model,
        initial_sequence=last_sequence,
        steps_ahead=forecast_horizon,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        feature_cols=feature_cols,
        df_history=df_lstm_clean
    )
    
    print(f"Generated LSTM forecast for {len(lstm_forecasts)} hours")
    print(f"Forecast period: {lstm_forecast_dates[0]} to {lstm_forecast_dates[-1]}")
    print(f"Forecast range: {lstm_forecasts.min():.1f} to {lstm_forecasts.max():.1f} μg/m³")
    print(f"Average forecast: {lstm_forecasts.mean():.1f} μg/m³")
    print(f"Standard deviation: {lstm_forecasts.std():.1f} μg/m³")
    
    # Create forecast visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Historical + Forecast
    historical_days = 14
    historical_hours = historical_days * 24
    hist_start_idx = max(0, len(df_lstm_clean) - historical_hours)
    
    hist_dates = df_lstm_clean['time'].iloc[hist_start_idx:]
    hist_values = df_lstm_clean['pm2_5 (μg/m³)'].iloc[hist_start_idx:]
    
    ax1.plot(hist_dates, hist_values, 'b-', linewidth=2, label='Historical PM2.5', alpha=0.8)
    ax1.plot(lstm_forecast_dates, lstm_forecasts, 'r-', linewidth=3, label='LSTM 7-Day Forecast', alpha=0.9)
    
    # Add confidence interval
    confidence_level = test_metrics['rmse']  # Use test RMSE
    ax1.fill_between(lstm_forecast_dates, 
                      lstm_forecasts - confidence_level, 
                      lstm_forecasts + confidence_level, 
                      color='red', alpha=0.2, label='Confidence Interval')
    
    ax1.axvline(x=lstm_forecast_dates[0], color='black', linestyle='--', linewidth=1.5)
    ax1.text(lstm_forecast_dates[0], ax1.get_ylim()[1]*0.95, 'Forecast Start', 
             rotation=90, verticalalignment='top', fontsize=10)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PM2.5 (μg/m³)')
    ax1.set_title('LSTM: PM2.5 Historical Data and 7-Day Forecast', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Forecast details
    ax2.bar(range(len(lstm_forecasts)), lstm_forecasts, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Hours into Future')
    ax2.set_ylabel('PM2.5 (μg/m³)')
    ax2.set_title('LSTM: Hourly Forecast for Next 7 Days')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal lines for health guidelines
    ax2.axhline(y=12, color='green', linestyle='--', alpha=0.5, label='Good (12 μg/m³)')
    ax2.axhline(y=35, color='orange', linestyle='--', alpha=0.5, label='Moderate (35 μg/m³)')
    ax2.legend()
    
    plt.tight_layout()
    forecast_plot_path = os.path.join(MODEL_DIR, 'lstm_weekly_forecast.png')
    plt.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save forecast to CSV
    forecast_df = pd.DataFrame({
        'timestamp': lstm_forecast_dates,
        'pm2_5_forecast': lstm_forecasts,
        'pm2_5_lower_bound': lstm_forecasts - confidence_level,
        'pm2_5_upper_bound': lstm_forecasts + confidence_level
    })
    
    forecast_csv_path = os.path.join(MODEL_DIR, 'lstm_pm25_7day_forecast.csv')
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"\nLSTM forecast saved to: {forecast_csv_path}")
    
except Exception as e:
    print(f"Error generating LSTM forecast: {e}")
    print("Creating simple forecast for visualization...")
    
    # Create a simple forecast based on historical patterns
    last_value = df_lstm_clean['pm2_5 (μg/m³)'].iloc[-1]
    lstm_forecasts = np.full(168, last_value)  # Constant forecast
    lstm_forecast_dates = [df_lstm_clean['time'].iloc[-1] + timedelta(hours=i+1) for i in range(168)]

# ==================== CREATE CONSOLIDATED ARTIFACTS FILE ====================
print("\n" + "="*50)
print("SAVING MODEL AND ARTIFACTS")
print("="*50)

# Create a comprehensive dictionary with all model artifacts
model_artifacts = {
    'model': model,
    'scaler_features': scaler_features,
    'scaler_target': scaler_target,
    'feature_cols': feature_cols,
    'model_params': {
        'sequence_length': SEQUENCE_LENGTH,
        'forecast_horizon': FORECAST_HORIZON,
        'input_shape': INPUT_SHAPE,
        'n_features': N_FEATURES
    },
    'test_metrics': test_metrics,
    'training_metrics': train_metrics,
    'validation_metrics': val_metrics,
    'model_summary': model.summary(),
    'training_history': history.history if 'history' in locals() else None,
    'forecast_functions': {
        'forecast_pm25_lstm': forecast_pm25_lstm,
        'multi_step_forecast_lstm': multi_step_forecast_lstm
    },
    'metadata': {
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'final_training_data_all_regions.csv',
        'region': 'doha' if 'region' in df.columns else 'all',
        'target_column': 'pm2_5 (μg/m³)'
    }
}

# Save consolidated artifacts as a single pickle file
consolidated_path = os.path.join(MODEL_DIR, 'lstm_model_artifacts.pkl')
with open(consolidated_path, 'wb') as f:
    pickle.dump(model_artifacts, f)
print(f"✓ Consolidated artifacts saved to: {consolidated_path}")

# Save metadata and metrics as JSON
metadata = {
    'model_params': model_artifacts['model_params'],
    'test_metrics': test_metrics,
    'training_metrics': train_metrics,
    'validation_metrics': val_metrics,
    'metadata': model_artifacts['metadata'],
    'feature_columns': feature_cols,
    'sequence_length': SEQUENCE_LENGTH,
    'forecast_horizon': FORECAST_HORIZON
}

metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4, default=str)

# Save training history
if 'history' in locals():
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(MODEL_DIR, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)

print("\n" + "="*50)
print("FILES SAVED IN 'LSTM_model' DIRECTORY")
print("="*50)
print(f"1. lstm_model_artifacts.pkl - Consolidated file with all artifacts")
print(f"2. model_metadata.json - Model parameters and metrics")
print(f"3. lstm_results.png - Model performance plots")
print(f"4. lstm_weekly_forecast.png - Weekly forecast visualization")
if 'forecast_df' in locals():
    print(f"9. lstm_pm25_7day_forecast.csv - Forecast data")
if 'history' in locals():
    print(f"10. training_history.csv - Training history")
print(f"11. best_lstm_model.keras - Best model from checkpoint")

print("\n" + "="*50)
print("HOW TO LOAD THE CONSOLIDATED MODEL ARTIFACTS")
print("="*50)
print("""
# Load all artifacts at once:
import pickle

with open('LSTM_model/lstm_model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler_features = artifacts['scaler_features']
scaler_target = artifacts['scaler_target']
feature_cols = artifacts['feature_cols']
model_params = artifacts['model_params']
test_metrics = artifacts['test_metrics']

# Use forecast function:
forecast_func = artifacts['forecast_functions']['forecast_pm25_lstm']
prediction = forecast_func(model, scaler_features, scaler_target, 
                          recent_data, feature_cols)
""")

print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(f"Sequence Length: {SEQUENCE_LENGTH} hours")
print(f"Forecast Horizon: {FORECAST_HORIZON} hour(s)")
print(f"Test R² Score: {test_metrics['r2']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.2f} μg/m³")
print(f"Test RMSE: {test_metrics['rmse']:.2f} μg/m³")
print("="*50)
