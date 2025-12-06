import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class LSTMModel(nn.Module):
    """LSTM model for multi-step time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layers
        self.fc_pm25 = nn.Linear(hidden_size, output_size)
        self.fc_pm10 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last time step's output
        out = self.dropout(out[:, -1, :])
        
        # Predict both PM2.5 and PM10
        pm25_pred = self.fc_pm25(out)
        pm10_pred = self.fc_pm10(out)
        
        return torch.cat([pm25_pred, pm10_pred], dim=1)

class AirQualityPredictor:
    """Main class for air quality prediction"""
    def __init__(self, lookback=168, forecast_horizon=720, region=None):
        """
        Args:
            lookback: Number of hours to look back (default: 1 week = 168 hours)
            forecast_horizon: Number of hours to predict (default: 1 month = 720 hours)
            region: If None, train separate models for each region
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.region = region
        self.scalers = {}
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_features(self, df):
        """Feature engineering and preprocessing"""
        df = df.copy()
        
        # Convert time and set as index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Sort by time
        df = df.sort_index()
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Lag features for targets
        df['pm2_5_lag_24h'] = df['pm2_5'].shift(24)
        df['pm10_lag_24h'] = df['pm10'].shift(24)
        df['pm2_5_lag_168h'] = df['pm2_5'].shift(168)
        df['pm10_lag_168h'] = df['pm10'].shift(168)
        
        # Rolling statistics
        df['pm2_5_rolling_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
        df['pm10_rolling_24h'] = df['pm10'].rolling(window=24, min_periods=1).mean()
        df['pm2_5_rolling_7d'] = df['pm2_5'].rolling(window=168, min_periods=1).mean()
        df['pm10_rolling_7d'] = df['pm10'].rolling(window=168, min_periods=1).mean()
        
        # Weather interaction features
        df['wind_dust_interaction'] = df['wind_speed_10m (km/h)'] * df['dust (μg/m³)']
        df['temp_humidity_interaction'] = df['temperature_2m (°C)'] * df['relative_humidity_2m (%)']
        
        # Fill missing values (from lag features)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def create_sequences(self, data, targets):
        """Create sequences for LSTM input"""
        sequences = []
        target_sequences = []
        
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            # Input sequence (lookback hours)
            seq = data[i:(i + self.lookback)]
            
            # Target sequence (next forecast_horizon hours)
            # For multi-step prediction, we predict future values
            target_seq = targets[(i + self.lookback):(i + self.lookback + self.forecast_horizon)]
            
            sequences.append(seq)
            target_sequences.append(target_seq)
        
        return np.array(sequences), np.array(target_sequences)
    
    def prepare_data(self, df, target_cols=['pm2_5', 'pm10']):
        """Prepare data for training"""
        # Feature engineering
        df_processed = self.prepare_features(df)
        
        # Select features
        feature_cols = [
            'temperature_2m (°C)', 'relative_humidity_2m (%)',
            'wind_speed_10m (km/h)', 'wind_direction_10m (°)',
            'precipitation (mm)', 'carbon_monoxide (μg/m³)',
            'nitrogen_dioxide (μg/m³)', 'sulphur_dioxide (μg/m³)',
            'ozone (μg/m³)', 'aerosol_optical_depth ()',
            'dust (μg/m³)', 'hour_sin', 'hour_cos', 'day_sin',
            'day_cos', 'month_sin', 'month_cos',
            'pm2_5_lag_24h', 'pm10_lag_24h', 'pm2_5_lag_168h',
            'pm10_lag_168h', 'pm2_5_rolling_24h', 'pm10_rolling_24h',
            'pm2_5_rolling_7d', 'pm10_rolling_7d',
            'wind_dust_interaction', 'temp_humidity_interaction'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df_processed.columns]
        
        # Scale features
        X = df_processed[available_features].values
        y = df_processed[target_cols].values
        
        # Fit scalers
        self.scalers['X'] = StandardScaler()
        self.scalers['y'] = MinMaxScaler()
        
        X_scaled = self.scalers['X'].fit_transform(X)
        y_scaled = self.scalers['y'].fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq, available_features
    
    def train(self, df, region_name, epochs=100, batch_size=32, learning_rate=0.001):
        """Train LSTM model for a specific region"""
        print(f"Training model for region: {region_name}")
        
        # Prepare data
        X_seq, y_seq, feature_names = self.prepare_data(df)
        
        # Split data (80% train, 10% validation, 10% test)
        n_samples = len(X_seq)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        X_train, y_train = X_seq[:train_size], y_seq[:train_size]
        X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
        X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
        
        # Create data loaders
        train_dataset = AirQualityDataset(X_train, y_train)
        val_dataset = AirQualityDataset(X_val, y_val)
        test_dataset = AirQualityDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=self.forecast_horizon,  # Predict horizon steps for each target
            dropout=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        # Store model
        self.models[region_name] = {
            'model': model,
            'feature_names': feature_names,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Evaluate on test set
        test_predictions, test_targets = self.evaluate_model(model, test_loader)
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, data_loader):
        """Evaluate model performance"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform predictions
        predictions_3d = predictions.reshape(-1, 2)
        targets_3d = targets.reshape(-1, 2)
        
        predictions_inv = self.scalers['y'].inverse_transform(predictions_3d)
        targets_inv = self.scalers['y'].inverse_transform(targets_3d)
        
        # Calculate metrics for first hour prediction
        pm25_pred_first = predictions_inv[::self.forecast_horizon, 0]
        pm25_true_first = targets_inv[::self.forecast_horizon, 0]
        pm10_pred_first = predictions_inv[::self.forecast_horizon, 1]
        pm10_true_first = targets_inv[::self.forecast_horizon, 1]
        
        print(f"PM2.5 RMSE: {np.sqrt(mean_squared_error(pm25_true_first, pm25_pred_first)):.2f}")
        print(f"PM2.5 MAE: {mean_absolute_error(pm25_true_first, pm25_pred_first):.2f}")
        print(f"PM10 RMSE: {np.sqrt(mean_squared_error(pm10_true_first, pm10_pred_first)):.2f}")
        print(f"PM10 MAE: {mean_absolute_error(pm10_true_first, pm10_pred_first):.2f}")
        
        return predictions, targets
    
    def predict_future(self, df, region_name, steps_ahead=720):
        """Predict future values for a region"""
        if region_name not in self.models:
            raise ValueError(f"No model trained for region: {region_name}")
        
        model = self.models[region_name]['model']
        model.eval()
        
        # Prepare the last lookback hours from data
        df_processed = self.prepare_features(df)
        feature_cols = self.models[region_name]['feature_names']
        
        # Get the last lookback hours
        last_sequence = df_processed[feature_cols].iloc[-self.lookback:].values
        last_sequence_scaled = self.scalers['X'].transform(last_sequence)
        
        # Reshape for LSTM (1, lookback, n_features)
        input_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(input_seq)
        
        # Reshape and inverse transform
        prediction_scaled_np = prediction_scaled.cpu().numpy().reshape(-1, 2)
        prediction = self.scalers['y'].inverse_transform(prediction_scaled_np)
        
        # Create future timestamps
        last_timestamp = df_processed.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=steps_ahead,
            freq='H'
        )
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': future_timestamps,
            'pm2_5_pred': prediction[:, 0],
            'pm10_pred': prediction[:, 1]
        })
        
        return result_df
    
    def plot_predictions(self, predictions_df, actual_df=None, region_name=""):
        """Plot predictions vs actuals"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # PM2.5 plot
        axes[0].plot(predictions_df['timestamp'], predictions_df['pm2_5_pred'], 
                    label='Predicted PM2.5', color='blue', linewidth=2)
        if actual_df is not None:
            axes[0].plot(actual_df['timestamp'], actual_df['pm2_5'], 
                        label='Actual PM2.5', color='red', alpha=0.7)
        axes[0].set_ylabel('PM2.5 (μg/m³)')
        axes[0].set_title(f'{region_name} - PM2.5 Predictions (Next Month)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PM10 plot
        axes[1].plot(predictions_df['timestamp'], predictions_df['pm10_pred'], 
                    label='Predicted PM10', color='green', linewidth=2)
        if actual_df is not None:
            axes[1].plot(actual_df['timestamp'], actual_df['pm10'], 
                        label='Actual PM10', color='orange', alpha=0.7)
        axes[1].set_ylabel('PM10 (μg/m³)')
        axes[1].set_xlabel('Timestamp')
        axes[1].set_title(f'{region_name} - PM10 Predictions (Next Month)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Example usage with your sample data
    df = pd.read_csv('your_data.csv')
    
    # Initialize predictor (predict next month = 720 hours)
    predictor = AirQualityPredictor(
        lookback=168,  # Look back 1 week
        forecast_horizon=720  # Predict 1 month ahead
    )
    
    # Train for each region
    regions = df['region'].unique()
    
    for region in regions:
        print(f"\n{'='*50}")
        print(f"Processing region: {region}")
        print(f"{'='*50}")
        
        # Filter data for region
        region_data = df[df['region'] == region].copy()
        
        # Split into train and test (last month for testing)
        train_data = region_data[region_data['time'] < '2024-12-01']
        test_data = region_data[region_data['time'] >= '2024-12-01']
        
        if len(train_data) > 1000:  # Ensure enough data
            # Train model
            model, train_loss, val_loss = predictor.train(
                train_data, 
                region_name=region,
                epochs=50,
                batch_size=32,
                learning_rate=0.001
            )
            
            # Make predictions for next month
            future_predictions = predictor.predict_future(
                train_data, 
                region_name=region,
                steps_ahead=720
            )
            
            # Plot predictions
            predictor.plot_predictions(
                future_predictions,
                actual_df=test_data if len(test_data) > 0 else None,
                region_name=region
            )
        else:
            print(f"Insufficient data for region {region}. Skipping...")