import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and clean data
dataset = "rows"
file_path = 'data/' + dataset + '.csv'
data = pd.read_csv(file_path)

# Convert timestamps to datetime
data['rpi_timestamp'] = pd.to_datetime(data['rpi_timestamp'], errors='coerce')
data['drone_timestamp'] = pd.to_datetime(data['drone_timestamp'], errors='coerce')

# Replace invalid values
data.replace([-999, '-999dB', '-999dBm'], np.nan, inplace=True)

data.dropna(inplace=True)

data['altitudeAMSL'] = data['altitudeAMSL'] * 1000

data.to_csv('data/cleaned_' + dataset + '.csv', index=False)

# Feature engineering
data['distance_from_center'] = np.sqrt(data['localPosition.x']**2 + data['localPosition.y']**2)
data['altitude_diff'] = data['altitudeAMSL'] - data['altitudeAMSL'].mean()
# Convert signal columns to numeric by stripping non-numeric characters
for col in ['rsrq', 'rsrp', 'rssi', 'sinr']:
    data[col] = pd.to_numeric(data[col].astype(str).str.extract(r'(-?\d+\.?\d*)')[0], errors='coerce')

# Compute the average signal quality
data['avg_signal_quality'] = data[['rsrq', 'rsrp', 'rssi', 'sinr']].mean(axis=1)
data['snr_improvement'] = data['sinr'] - data['rssi']

features = ['localPosition.x', 'localPosition.y', 'altitudeAMSL', 'distance_from_center', 'altitude_diff', 'avg_signal_quality', 'snr_improvement']
target = 'rssi'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lr = make_pipeline(StandardScaler(), LinearRegression())

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Generate six figures
plt.figure(figsize=(10, 6))
sns.histplot(data['rssi'], bins=50, kde=True, color='skyblue')
plt.title('Histogram of RSSI Values')
plt.xlabel('RSSI (dBm)')
plt.ylabel('Frequency')
plt.savefig('fig1_rssi_hist.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['distance_from_center'], y=data['rssi'])
plt.title('Distance from Center vs RSSI')
plt.xlabel('Distance from Center (m)')
plt.ylabel('RSSI (dBm)')
plt.savefig('fig2_distance_rssi.png')

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('fig3_correlation.png')

plt.figure(figsize=(10, 6))
y_pred_rf = rf.predict(X_test)
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.title('Random Forest Predictions vs Actual')
plt.xlabel('Actual RSSI')
plt.ylabel('Predicted RSSI')
plt.savefig('fig4_rf_predictions.png')

plt.figure(figsize=(10, 6))
y_pred_gb = gb.predict(X_test)
sns.scatterplot(x=y_test, y=y_pred_gb)
plt.title('Gradient Boosting Predictions vs Actual')
plt.xlabel('Actual RSSI')
plt.ylabel('Predicted RSSI')
plt.savefig('fig5_gb_predictions.png')

plt.figure(figsize=(10, 6))
plt.tricontourf(data['localPosition.x'], data['localPosition.y'], data['rssi'], levels=60, cmap='coolwarm')
plt.colorbar(label='RSSI (dBm)')
plt.title('Predicted Signal Strength Map')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.savefig('fig6_signal_map.png')

plt.show()