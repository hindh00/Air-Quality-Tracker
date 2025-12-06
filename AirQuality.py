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

# Check GPU availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class AirQualityDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return tensors directly (they'll be moved to GPU in training)
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class LSTMModel(nn.Module):
    """LSTM model for multi-step time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers, output_steps, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
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
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_steps * 2)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden and cell states on the same device as x
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last time step's output
        out = self.dropout(out[:, -1, :])
        
        # Predict all time steps for both targets
        predictions = self.fc(out)
        
        # Reshape to (batch_size, output_steps, 2)
        predictions = predictions.view(batch_size, self.output_steps, 2)
        
        return predictions

class AirQualityPredictor:
    """Main class for air quality prediction with GPU acceleration"""
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
        self.device = device  # Use the global device
    
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
        df['pm2_5_lag_24h'] = df['pm2_5 (μg/m³)'].shift(24)
        df['pm10_lag_24h'] = df['pm10 (μg/m³)'].shift(24)
        df['pm2_5_lag_168h'] = df['pm2_5 (μg/m³)'].shift(168)
        df['pm10_lag_168h'] = df['pm10 (μg/m³)'].shift(168)
        
        # Rolling statistics
        df['pm2_5_rolling_24h'] = df['pm2_5 (μg/m³)'].rolling(window=24, min_periods=1).mean()
        df['pm10_rolling_24h'] = df['pm10 (μg/m³)'].rolling(window=24, min_periods=1).mean()
        df['pm2_5_rolling_7d'] = df['pm2_5 (μg/m³)'].rolling(window=168, min_periods=1).mean()
        df['pm10_rolling_7d'] = df['pm10 (μg/m³)'].rolling(window=168, min_periods=1).mean()
        
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
            target_seq = targets[(i + self.lookback):(i + self.lookback + self.forecast_horizon)]
            
            sequences.append(seq)
            target_sequences.append(target_seq)
        
        return np.array(sequences), np.array(target_sequences)
    
    def prepare_data(self, df, target_cols=['pm2_5 (μg/m³)', 'pm10 (μg/m³)']):
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
        
        print(f"Created {len(X_seq)} sequences")
        print(f"X shape: {X_seq.shape}")
        print(f"y shape: {y_seq.shape}")
        
        return X_seq, y_seq, available_features
    
    def train(self, df, region_name, epochs=100, batch_size=32, learning_rate=0.001):
        """Train LSTM model for a specific region with GPU acceleration"""
        print(f"Training model for region: {region_name}")
        print(f"Using device: {self.device}")
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Prepare data
        X_seq, y_seq, feature_names = self.prepare_data(df)
        
        # Split data (80% train, 10% validation, 10% test)
        n_samples = len(X_seq)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        X_train, y_train = X_seq[:train_size], y_seq[:train_size]
        X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
        X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
        
        # Create data loaders with GPU optimization
        train_dataset = AirQualityDataset(X_train, y_train)
        val_dataset = AirQualityDataset(X_val, y_val)
        test_dataset = AirQualityDataset(X_test, y_test)
        
        # Optimize batch size for GPU
        if torch.cuda.is_available():
            batch_size = min(64, batch_size * 2)  # Larger batches for GPU
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,  # Faster data transfer to GPU
            num_workers=2     # Parallel data loading
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        
        # Initialize model and move to GPU
        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_steps=self.forecast_horizon,
            dropout=0.3
        ).to(self.device)  # This moves the model to GPU
        
        print(f"Model architecture:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: 128")
        print(f"  Output shape: (batch_size, {self.forecast_horizon}, 2)")
        print(f"  Model moved to: {next(model.parameters()).device}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop with GPU acceleration
        train_losses = []
        val_losses = []
        
        import time
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Move batch to GPU
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                
                # Debug print for first batch of first epoch
                if epoch == 0 and train_loss == 0:
                    print(f"Batch X device: {batch_X.device}")
                    print(f"Batch y device: {batch_y.device}")
                    print(f"Model device: {next(model.parameters()).device}")
                    print(f"Batch X shape: {batch_X.shape}")
                    print(f"Batch y shape: {batch_y.shape}")
                    print(f"Output shape: {outputs.shape}")
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
            
            epoch_time = time.time() - epoch_start_time
            
            if (epoch + 1) % 10 == 0:
                # Print GPU memory usage if using GPU
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f'Epoch [{epoch+1}/{epochs}] - Time: {epoch_time:.2f}s - '
                          f'Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - '
                          f'GPU Mem: {memory_allocated:.2f}/{memory_reserved:.2f} GB')
                else:
                    print(f'Epoch [{epoch+1}/{epochs}] - Time: {epoch_time:.2f}s - '
                          f'Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
        
        total_training_time = time.time() - total_start_time
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Average time per epoch: {total_training_time/epochs:.2f} seconds")
        
        # Store model
        self.models[region_name] = {
            'model': model,
            'feature_names': feature_names,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
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
                
                # Move predictions back to CPU for numpy conversion
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform predictions
        predictions_2d = predictions.reshape(-1, 2)
        targets_2d = targets.reshape(-1, 2)
        
        predictions_inv = self.scalers['y'].inverse_transform(predictions_2d)
        targets_inv = self.scalers['y'].inverse_transform(targets_2d)
        
        # Reshape back
        n_samples = predictions.shape[0]
        predictions_inv = predictions_inv.reshape(n_samples, self.forecast_horizon, 2)
        targets_inv = targets_inv.reshape(n_samples, self.forecast_horizon, 2)
        
        # Calculate metrics for first hour prediction
        pm25_pred_first = predictions_inv[:, 0, 0]
        pm25_true_first = targets_inv[:, 0, 0]
        pm10_pred_first = predictions_inv[:, 0, 1]
        pm10_true_first = targets_inv[:, 0, 1]
        
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
        
        # Reshape for LSTM (1, lookback, n_features) and move to GPU
        input_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(input_seq)
        
        # Move prediction back to CPU
        prediction_scaled_np = prediction_scaled.cpu().numpy().reshape(-1, 2)
        prediction = self.scalers['y'].inverse_transform(prediction_scaled_np)
        
        # Reshape to (steps_ahead, 2)
        prediction = prediction.reshape(steps_ahead, 2)
        
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
        """Plot predictions vs actuals with fixed datetime handling"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert predictions_df timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(predictions_df['timestamp']):
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # PM2.5 plot
        axes[0].plot(predictions_df['timestamp'], predictions_df['pm2_5_pred'], 
                    label='Predicted PM2.5', color='blue', linewidth=2)
        
        if actual_df is not None:
            # Ensure actual_df['time'] is datetime
            if 'time' in actual_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(actual_df['time']):
                    actual_df['time'] = pd.to_datetime(actual_df['time'])
                
                # Filter actual data to match prediction timeframe
                mask = (actual_df['time'] >= predictions_df['timestamp'].iloc[0]) & \
                    (actual_df['time'] <= predictions_df['timestamp'].iloc[-1])
                actual_filtered = actual_df[mask]
                
                if len(actual_filtered) > 0:
                    axes[0].plot(actual_filtered['time'], actual_filtered['pm2_5 (μg/m³)'], 
                                label='Actual PM2.5', color='red', alpha=0.7, linewidth=1)
        
        axes[0].set_ylabel('PM2.5 (μg/m³)')
        axes[0].set_title(f'{region_name} - PM2.5 Predictions (Next {len(predictions_df)} hours)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Format x-axis for PM2.5 plot
        if len(predictions_df) > 24:
            # If more than 1 day, show date and time
            axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
        else:
            # If less than 1 day, show only time
            axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # PM10 plot
        axes[1].plot(predictions_df['timestamp'], predictions_df['pm10_pred'], 
                    label='Predicted PM10', color='green', linewidth=2)
        
        if actual_df is not None and 'time' in actual_df.columns:
            if len(actual_filtered) > 0:
                axes[1].plot(actual_filtered['time'], actual_filtered['pm10 (μg/m³)'], 
                            label='Actual PM10', color='orange', alpha=0.7, linewidth=1)
        
        axes[1].set_ylabel('PM10 (μg/m³)')
        axes[1].set_xlabel('Timestamp')
        axes[1].set_title(f'{region_name} - PM10 Predictions (Next {len(predictions_df)} hours)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Format x-axis for PM10 plot
        if len(predictions_df) > 24:
            axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
        else:
            axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'predictions_{region_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, region_name, path):
        """Save trained model"""
        if region_name in self.models:
            torch.save({
                'model_state_dict': self.models[region_name]['model'].state_dict(),
                'feature_names': self.models[region_name]['feature_names'],
                'scalers': self.scalers,
                'lookback': self.lookback,
                'forecast_horizon': self.forecast_horizon
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, region_name, path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model
        input_size = len(checkpoint['feature_names'])
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_steps=checkpoint['forecast_horizon'],
            dropout=0.3
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.models[region_name] = {
            'model': model,
            'feature_names': checkpoint['feature_names']
        }
        self.scalers = checkpoint['scalers']
        self.lookback = checkpoint['lookback']
        self.forecast_horizon = checkpoint['forecast_horizon']
        
        print(f"Model loaded from {path}")

# Main execution with GPU optimization
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('final_training_data_all_regions.csv')
    
    print("="*60)
    print("GPU ACCELERATED AIR QUALITY PREDICTION")
    print("="*60)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print("✓ GPU acceleration enabled")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    else:
        print("✗ GPU not available, using CPU")
        print("Note: Training will be slower on CPU")
    
    # Start with smaller parameters for testing
    print("\nStarting with test configuration...")
    predictor = AirQualityPredictor(
        lookback=24,    # Reduced for faster testing
        forecast_horizon=24  # Reduced for faster testing
    )
    
    # Test with first region
    regions = df['region'].unique()
    
    for region in regions[:1]:  # Only process first region for testing
        print(f"\n{'='*50}")
        print(f"Processing region: {region}")
        print(f"{'='*50}")
        
        # Filter data for region
        region_data = df[df['region'] == region].copy()
        
        # Check data size
        min_samples_needed = predictor.lookback + predictor.forecast_horizon + 100
        if len(region_data) < min_samples_needed:
            print(f"Warning: Insufficient data for {region}")
            print(f"Need at least {min_samples_needed} samples, have {len(region_data)}")
            continue
        
        print(f"Data points: {len(region_data)}")
        
        # Split data
        split_idx = int(0.8 * len(region_data))
        train_data = region_data.iloc[:split_idx]
        test_data = region_data.iloc[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Train model with GPU acceleration
        model, train_loss, val_loss = predictor.train(
            train_data, 
            region_name=region,
            epochs=30,  # Reduced epochs for testing
            batch_size=32,
            learning_rate=0.001
        )
        
        # Make predictions
        print("\nMaking predictions...")
        future_predictions = predictor.predict_future(
            train_data, 
            region_name=region,
            steps_ahead=min(168, predictor.forecast_horizon)  # Predict up to 1 week
        )
        
        # Plot predictions
        predictor.plot_predictions(
            future_predictions,
            actual_df=test_data if len(test_data) > 0 else None,
            region_name=region
        )
        
        # Save model
        predictor.save_model(region, f'model_{region}.pth')
        
        print(f"\nCompleted processing for {region}")
        break  # Only process first region for testing
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)