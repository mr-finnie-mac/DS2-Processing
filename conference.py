import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load cleaned dataset
file_path = 'data/cleaned_rows.csv'  # Adjust path if needed
data = pd.read_csv(file_path)

# Convert timestamps to datetime
for col in ['rpi_timestamp', 'drone_timestamp']:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# Feature Engineering
# Calculate altitude difference from mean altitude
mean_altitude = data['altitudeAMSL'].mean()
data['altitude_diff'] = data['altitudeAMSL'] - mean_altitude

# Calculate distance from center using Euclidean distance
ref_x, ref_y = data['localPosition.x'].mean(), data['localPosition.y'].mean()
data['distance_from_center'] = np.sqrt((data['localPosition.x'] - ref_x)**2 + (data['localPosition.y'] - ref_y)**2)

# Additional Features
# Humidity squared
if 'humidity' in data.columns:
    data['humidity_squared'] = data['humidity']**2

# Signal Quality Metric (Mean of all signal fields)
if {'rsrq', 'rsrp', 'rssi', 'sinr'}.issubset(data.columns):
    data['avg_signal_quality'] = data[['rsrq', 'rsrp', 'rssi', 'sinr']].mean(axis=1)
    data['snr_improvement'] = data['sinr'] - data['rssi']

# Define feature set based on cleaned dataset format
features = ['localPosition.x', 'localPosition.y', 'altitudeAMSL', 'distance_from_center',
            'altitude_diff', 'humidity_squared', 'avg_signal_quality', 'snr_improvement']
target = 'rssi'  # Target variable

# Ensure selected features exist
features = [f for f in features if f in data.columns]
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lr = make_pipeline(StandardScaler(), LinearRegression())

# Train models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Evaluate models
models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Linear Regression': lr}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Generate Signal Strength Map
x_grid, y_grid = np.meshgrid(
    np.linspace(data['localPosition.x'].min(), data['localPosition.x'].max(), 100),
    np.linspace(data['localPosition.y'].min(), data['localPosition.y'].max(), 100)
)

grid_data = pd.DataFrame({'localPosition.x': x_grid.ravel(), 'localPosition.y': y_grid.ravel()})
grid_data['altitudeAMSL'] = mean_altitude
grid_data['distance_from_center'] = np.sqrt((grid_data['localPosition.x'] - ref_x)**2 + (grid_data['localPosition.y'] - ref_y)**2)
grid_data['altitude_diff'] = grid_data['altitudeAMSL'] - mean_altitude
grid_data['humidity_squared'] = data['humidity_squared'].mean()
grid_data['avg_signal_quality'] = data[['rsrq', 'rsrp', 'rssi', 'sinr']].mean(axis=1).mean()
grid_data['snr_improvement'] = data['sinr'].mean() - data['rssi'].mean()

# Predict signal strength
if set(features).issubset(grid_data.columns):
    grid_features = grid_data[features]
    grid_data['rssi'] = (rf.predict(grid_features) + gb.predict(grid_features) + lr.predict(grid_features)) / 3

    # Visualize Signal Strength Map
    plt.figure(figsize=(10, 8))
    plt.tricontourf(grid_data['localPosition.x'], grid_data['localPosition.y'], grid_data['rssi'], levels=60, cmap='coolwarm')
    plt.colorbar(label='Signal Strength (dBm)')
    plt.title("Predicted Signal Strength Map (X, Y Coordinates)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.show()

else:
    print("Error: Grid features missing some required columns.")
