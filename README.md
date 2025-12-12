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

## Usage
### Web Dashboard
- Select a region from the sidebar dropdown
- View current air quality metrics
- Use the slider to explore 24-hour forecasts
- Observe the 3D map showing PM2.5 concentrations
- Analyze the forecast trend chart

## Model Development
### Training Process
- Data Collection: Historical data from Open-Meteo API (2023-2025)

- Feature Engineering: 19 features including cyclical time encoding

- Model Architecture: Bidirectional LSTM with attention mechanism

- Training: 50 epochs with early stopping and dropout regularization

- Validation: Hold-out test set with temporal cross-validation

## File Structure
Air-Quality-Tracker/
│
├── Dashboard/                 
│   ├── app.py                                 # Streamlit web dashboard (main application)
│   ├── universal_lstm_model.keras             # Pre-trained Bidirectional LSTM 
|   └── requirements.txt                       # Required dependencies
├── Data/
|   ├── final_test_data_all_regions.csv        # Training dataset
|   └── final_training_data_all_regions.csv    # Testing dataset
├── Data Visualization/
|   └── data_visualization.ipynb               # Initial Data Visualization
model
│       - Model weights and architecture
│       - Trained on 2023-2025 Qatar air quality data
│
├── Models/                             # All machine learning models
│   ├── LSTM/                           # Long Short-Term Memory implementations
│   │   ├── LSTM_0.1.ipynb              # Initial LSTM prototype
│   │   │   - Basic unidirectional LSTM
│   │   │   - Initial feature engineering
│   │   │   - Baseline performance testing
│   │   ├── LSTM_0.2.ipynb              # Improved LSTM with bidirectional layers
│   │   │   - Bidirectional LSTM implementation
│   │   │   - Attention mechanism
│   │   │   - Dropout regularization
│   │   └── LSTM_0.3.ipynb              # Final production model
│   │       - Hyperparameter optimization
│   │       - Cross-validation results
│   │       - Model saving/loading utilities
│   │
│   ├── SARIMA/                         # Statistical time series models
│   │   └── SARIMA_Model.ipynb          # Seasonal ARIMA implementation
│   │       - Univariate time series analysis
│   │       - Seasonality detection
│   │       - Statistical baseline comparison
│   │
│   ├── XGBoost/                        # Gradient boosting models
│   │   └── XGBoost_Model.ipynb         # XGBoost regression
│   │       - Feature importance analysis
│   │       - Hyperparameter tuning
│   │       - Comparison with deep learning models
│   │
│   └── CNN/                            # Convolutional Neural Network
│       └── CNN_Model.ipynb             # CNN for time series
│           - 1D convolutional layers
│           - Pattern recognition in temporal data
│           - Spatial feature extraction
│
├── Data/                               # Datasets and data processing
│   ├── final_training_data_all_regions.csv
│   │   - Historical data (2023-2025)
│   │   - All 6 municipalities
│   │   - 19 features including weather and pollution data
│   │
│   ├── raw_data/                       # Raw data from APIs (if cached)
│   │   ├── weather_data_*.json         # Open-Meteo weather responses
│   │   └── aq_data_*.json              # Air quality API responses
│   │
│   └── processed/                      # Cleaned and engineered datasets
│       ├── training_dataset.pkl        # Pickled training data
│       ├── test_dataset.pkl           # Hold-out test set
│       └── scalers.pkl                # Fitted MinMaxScaler objects
│
├── data_visualization.py               # Exploratory Data Analysis (EDA)
│   - Correlation matrix heatmaps
│   - Temporal trend analysis (2023-2025)
│   - Distribution plots
│   - Feature relationship visualizations
│   - Model performance charts
│
├── utils/                              # Utility functions
│   ├── __init__.py
│   ├── data_fetcher.py                # Open-Meteo API wrapper
│   │   - Cached API requests
│   │   - Data merging and cleaning
│   │   - Timezone handling
│   │
│   ├── feature_engineering.py         # Feature creation
│   │   - Cyclical time encoding
│   │   - Wind direction transformation
│   │   - One-hot city encoding
│   │
│   ├── model_utils.py                 # Model helpers
│   │   - Sequence generation for LSTM
│   │   - Forecast evaluation metrics
│   │   - Model serialization
│   │
│   └── visualization_utils.py         # Plotting functions
│       - EPA color coding for PM2.5
│       - Map layer creation for PyDeck
│       - Chart styling and formatting
│
├── tests/                              # Unit and integration tests
│   ├── test_data_fetcher.py           # API wrapper tests
│   ├── test_feature_engineering.py    # Feature creation tests
│   ├── test_model_predictions.py      # Model inference tests
│   └── test_dashboard.py              # Streamlit app tests
│
├── config/                             # Configuration files
│   ├── constants.py                    # Project constants
│   │   - LOCATIONS dictionary (lat/lon)
│   │   - EPA PM2.5 thresholds
│   │   - Feature column names
│   │
│   └── settings.yaml                   # Environment settings
│       - API endpoints
│       - Model parameters
│       - Visualization settings
│
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── EDA_Complete.ipynb             # Comprehensive EDA
│   ├── Model_Comparison.ipynb         # All models side-by-side
│   └── Feature_Importance.ipynb       # Feature contribution analysis
│
├── assets/                             # Images and media
│   ├── architecture_diagram.png       # System architecture
│   ├── correlation_heatmap.png        # Feature correlation
│   ├── performance_charts/            # Model performance visuals
│   │   ├── training_history.png
│   │   ├── predictions_vs_actual.png
│   │   └── error_distribution.png
│   └── screenshots/                   # Dashboard screenshots
│       ├── dashboard_home.png
│       ├── forecast_map.png
│       └── data_table.png
│
├── requirements.txt                    # Python dependencies
│   - Streamlit==1.28.0
│   - tensorflow==2.13.0
│   - pandas==2.0.3
│   - numpy==1.24.3
│   - openmeteo-requests==1.1.1
│   - requests-cache==1.1.0
│   - pydeck==0.8.1b0
│   - scikit-learn==1.3.0
│   - matplotlib==3.7.2
│   - seaborn==0.12.2
│
├── .streamlit/                         # Streamlit configuration
│   └── config.toml                     # Streamlit app settings
│       - Theme configuration
│       - Layout settings
│       - Caching parameters
│
├── .cache/                             # API response cache (auto-generated)
│   └── __pycache__/
│
├── environment.yml                     # Conda environment specification
│   - Alternative to requirements.txt
│   - For reproducible environments
│
├── Dockerfile                          # Containerization configuration
│   - For deploying as Docker container
│   - Includes all dependencies
│
├── docker-compose.yml                  # Multi-container orchestration
│   - App + database (if added later)
│   - Volume mapping
│
├── LICENSE                             # MIT License file
│
├── README.md                           # This documentation file
│
└── report/                             # Project documentation
    ├── Project_Report.pdf             # Complete project report
    ├── presentation.pptx              # Team presentation slides
    └── technical_documentation.md     # Detailed technical specs
