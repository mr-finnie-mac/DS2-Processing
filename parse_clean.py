import pandas as pd
import numpy as np


# PARSE AND CLEAN DATA

# load the dataset 
dataset = "rows"
file_path = 'data/' + dataset + '.csv'
data = pd.read_csv(file_path)

print("Initial Data Overview:")
print(data.info())
print(data.head())

# convert timestamps to datetime (had issue with the format)
for col in ['rpi_timestamp', 'drone_timestamp']:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')  # convert + handle errors

# handle missing or invalid values
# Replace bad values (-999, etc.) with nan
invalid_values = [-999, '-999dB', '-999dBm']
data.replace(invalid_values, np.nan, inplace=True)

# format signal field
signal_fields = ['rsrq', 'rsrp', 'rssi', 'sinr']
for field in signal_fields:
    if field in data.columns:
        data[field] = data[field].str.extract(r'(-?\d+\.?\d*)').astype(float) # remove dBm part from signal field

# normalise and clean fields
# amke sure all numerical columns are float
numeric_fields = ['temperature', 'pressure', 'humidity', 'gas_resistance', 'altitude',
                  'altitudeRelative', 'altitudeAMSL', 'gps.lat', 'gps.lon',
                  'localPosition.x', 'localPosition.y', 'localPosition.z', 'hdop', 'vdop']
for field in numeric_fields:
    if field in data.columns:
        data[field] = pd.to_numeric(data[field], errors='coerce')

# drop duplicates and handle NaNs
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)  # Drop rows with NaN values (not sure maybe should keep)

# normalise units if needed - convert altitude from meters to a consistent scale
# if 'altitudeAMSL' is in kilometers, convert to meters - CHECK!
data['altitudeAMSL'] = data['altitudeAMSL'] * 1000  # might need to adjust

# cleaned data
print("\nCleaned Data Overview:")
print(data.info())
print(data.head())

# save cleaned data to a new file
data.to_csv('data/cleaned_'+dataset+'.csv', index=False)
print("Cleaned data saved to 'cleaned_"+dataset+".csv'.")


# FEATURE ENGINEERING

import numpy as np

# create spatial features
# define a central reference point (e.g., average latitude and longitude) - this should be the position of antenna! (takeoff pose + 2 m north and east)
ref_lon = data['localPosition.x'].mean()
ref_lat = data['localPosition.y'].mean() # maybe should be x,y?

# calculate distance from reference point using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # distance in kilometers

data['distance_from_center'] = haversine(data['gps.lat'], data['gps.lon'], ref_lat, ref_lon)

# get altitude difference from a reference altitude (mean or min altitude)
data['altitude_diff'] = data['altitudeAMSL'] - data['altitudeAMSL'].mean()

# create signal quality features
# aggregate signal metric (average signal strength)
data['avg_signal_quality'] = data[['rsrq', 'rsrp', 'rssi', 'sinr']].mean(axis=1)

# make signal-to-noise ratio improvement feature
data['snr_improvement'] = data['sinr'] - data['rssi']

# create environmental features
# normalise pressure to sea level using barometric formula (simplified)
P0 = 1013.25  # standard atmospheric pressure at sea level in hPa
T0 = 288.15   # standard temp in kelvin
L = 0.0065    # temp lapse rate in K/m
R = 8.31447   # universal gas constant
g = 9.80665   # gravitational acceleration
M = 0.0289644 # molar mass of earths air

data['pressure_sea_level'] = data['pressure'] * (1 + (data['altitudeAMSL'] * L / T0))

# temporal features
# get hour and day of the week
data['hour_of_day'] = data['rpi_timestamp'].dt.hour
data['day_of_week'] = data['rpi_timestamp'].dt.dayofweek

#interaction and polynomial features
# create polynomial terms (square of humidity)
data['humidity_squared'] = data['humidity']**2

# interaction - temperature and humidity
data['temp_humidity_interaction'] = data['temperature'] * data['humidity']

# verify created features
print("New Features Overview:")
print(data[['distance_from_center', 'altitude_diff', 'avg_signal_quality', 
           'snr_improvement', 'pressure_sea_level', 'hour_of_day', 
           'day_of_week', 'humidity_squared', 'temp_humidity_interaction']].head())

# save data with new features
data.to_csv(dataset +'_features.csv', index=False)
print("Feature-engineered data saved to _" + dataset + "rows_features.csv'.")

# ANALYSING THE FEATURES

import matplotlib.pyplot as plt
import seaborn as sns

# statistical summaries
print("Statistical Summary of New Features:")
print(data[['distance_from_center', 'altitude_diff', 'avg_signal_quality', 
           'snr_improvement', 'pressure_sea_level', 'hour_of_day', 
           'day_of_week', 'humidity_squared', 'temp_humidity_interaction']].describe())

# visualize distributions
# plot histograms for continuous features
features_to_plot = ['distance_from_center', 'altitude_diff', 'avg_signal_quality', 
                    'snr_improvement', 'pressure_sea_level', 'humidity_squared', 
                    'temp_humidity_interaction']
data[features_to_plot].hist(bins=30, figsize=(15, 10), grid=False)
plt.suptitle('Histograms of Continuous Features', fontsize=16)
plt.show()

# convert drone_timestamp columns to Unix timestamps (seconds since epoch)
data['rpi_timestamp'] = pd.to_datetime(data['rpi_timestamp'])
data['drone_timestamp'] = pd.to_datetime(data['drone_timestamp'])

data['rpi_timestamp'] = data['rpi_timestamp'].astype(np.int64) // 10**9  # convert to Unix drone_timestamp (seconds)
data['drone_timestamp'] = data['drone_timestamp'].astype(np.int64) // 10**9  # convert to Unix drone_timestamp (seconds)

# compute the correlation matrix
correlation_matrix = data.corr()


# plot a heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f",  
    cmap='coolwarm', 
    cbar=True, 
    xticklabels=True, 
    yticklabels=True
)
plt.title("Correlation Matrix", fontsize=16)
plt.show()


# relationships with key targets
# scatterplots for relationships with `temperature` and `rsrq`
targets = ['temperature', 'rsrq']
for target in targets:
    for feature in features_to_plot:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=data[feature], y=data[target])
        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

# feature importance using a Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# prepare data for feature importance analysis
X = data[features_to_plot]
y = data['rssi']  # target variable

# test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train RF model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importances = pd.DataFrame({'Feature': features_to_plot, 
                                    'Importance': model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# plot feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# prepare the data
# get features and target variable for signal strength
features = ['localPosition.x', 'localPosition.y', 'altitudeAMSL', 'distance_from_center', 
            'altitude_diff', 'humidity_squared', 'avg_signal_quality', 'snr_improvement']
target = 'rssi'  # Replace with 'rsrq' or 'rsrp' if desired

X = data[features]
y = data[target]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define ensemble models
# model 1: Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42) # trimmed to stop overffintg


# model 2: Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# model 3: Linear Regression with scaling
lr = make_pipeline(StandardScaler(), LinearRegression())

# train models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# evaluate models
models = {'Random Forest': rf, 'Gradient Boosting': gb, 'Linear Regression': lr}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# combine predictions (stacking) + average predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
lr_pred = lr.predict(X_test)

ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3
# ensemble_pred = np.clip(ensemble_pred, -120, -50) # issues with signal being outside range!
print(ensemble_pred)

# eval ensemble model
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f"Ensemble Model - MAE: {ensemble_mae:.2f}, RMSE: {ensemble_rmse:.2f}")

# generate signal strength map using x and y coords
# define the range for x and y coords (based on data)
x_min, x_max = data['localPosition.x'].min(), data['localPosition.x'].max()
y_min, y_max = data['localPosition.y'].min(), data['localPosition.y'].max()

# create a grid of x and y coordinates
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

# predict signal strength for each grid point
grid_data = pd.DataFrame({'localPosition.x': x_grid.ravel(), 'localPosition.y': y_grid.ravel()})
grid_data['altitudeAMSL'] = data['altitudeAMSL'].mean()
grid_data['distance_from_center'] = np.sqrt(grid_data['localPosition.x']**2 + grid_data['localPosition.y']**2)
grid_data['altitude_diff'] = grid_data['altitudeAMSL'] - data['altitudeAMSL'].mean()
grid_data['humidity_squared'] = data['humidity_squared'].mean()
grid_data['temp_humidity_interaction'] = data['temp_humidity_interaction'].mean()
grid_data['avg_signal_quality'] = data[['rsrq', 'rsrp', 'rssi', 'sinr']].mean(axis=1)
grid_data['snr_improvement'] = data['sinr'] - data['rssi']

# check for nan values in grid_data before making predictions
if grid_data.isnull().values.any():
    print("NaN values found in grid_data! Filling NaNs...")
    grid_data.fillna(grid_data.mean(), inplace=True)  # maybe, fill NaN values with the column mean?
    # grid_data.dropna(inplace=True) # or drop them

# predict using ensemble model
grid_features = grid_data[features]
grid_data['rssi'] = (rf.predict(grid_features) + gb.predict(grid_features) + lr.predict(grid_features)) / 3
# grid_data['rssi'] = np.clip(grid_data['rssi'], -120, -50)


# visualize signal strength map
plt.figure(figsize=(10, 8))
plt.tricontourf(grid_data['localPosition.x'], grid_data['localPosition.y'], grid_data['rssi'], levels=60, cmap='coolwarm')
plt.colorbar(label='Signal Strength (dBm)')
plt.title("Predicted Signal Strength Map (x, y Coordinates)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.show()

# debug
# summary statistics of the target column 
print(data['rssi'].describe())

# visualize the distribution of rssi values
plt.hist(grid_data['rssi'], bins=50, color='skyblue', edgecolor='black')
plt.title('Signal Strength (RSSI) Distribution')
plt.xlabel('RSSI (dBm)')
plt.ylabel('Frequency')
plt.show()

# visualize signal strength map with flight path
plt.figure(figsize=(10, 8))

# create the signal strength map
plt.tricontourf(grid_data['localPosition.x'], grid_data['localPosition.y'], grid_data['rssi'], levels=60, cmap='coolwarm')
plt.colorbar(label='Signal Strength (dBm)')
plt.title("Predicted Signal Strength Map with Flight Path")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

flight_data = data[['localPosition.x', 'localPosition.y']]  # Replace with your actual flight path data

# Overlay the flight path
plt.plot(flight_data['localPosition.x'], flight_data['localPosition.y'], color='white', linestyle='-', linewidth=2, label="Flight Path")
# plt.scatter(flight_data['localPosition.x'], flight_data['localPosition.y'], color='white', label="Flight Path") # or plot as points
plt.legend()
plt.show()
