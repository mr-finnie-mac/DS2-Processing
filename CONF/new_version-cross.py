import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


# Example function for loading models
def load_pytorch_model(model_class, model_path):
    model = model_class(input_dim=10)  # Make sure input_dim is the same as during training
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to eval mode
    return model

def load_keras_model(model_path):
    return load_model(model_path)

# Example for data preprocessing (you can adjust based on your needs)
def preprocess_new_data(df, feature_columns, scaler=None):
    features = df[feature_columns]
    
    if scaler:
        features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    return torch.tensor(features, dtype=torch.float32), scaler

# Model evaluation function
def evaluate_model(model, X_new, y_true, model_type='pytorch'):
    if model_type == 'pytorch':
        with torch.no_grad():
            y_pred = model(X_new).numpy()
    elif model_type == 'keras':
        y_pred = model.predict(X_new)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return mse, mae



# Load all models (adjust paths accordingly)
models = {
    'unet': load_keras_model('unet_model.h5'),  # If using Keras for U-Net
    'transformer': load_pytorch_model(SignalQualityTransformer, 'transformer_model.pth'),
    # Add other models as needed (IDW, RIGGING, Ensemble)
}

# Load new dataset layer (e.g., 20m altitude dataset)
new_data = pd.read_csv('new_dataset_20m.csv')

# Feature engineering
# Tower distance
distances = compute_distance_to_tower(new_data, tower_location=tower_position)/1000
# train['distance_to_tower'] = distances
new_data.loc[:, 'distance_to_tower'] = distances
# Tower direction
_, elevation_angle = compute_tower_direction_local(new_data)
azimuths,_ = compute_tower_direction(new_data, tower_location=tower_position)
# train['azimuth'] = azimuths # add az to input
new_data.loc[:, 'azimuth'] = azimuths
# train['elevation_angle'] = elevation_angle # add e angle to input
new_data.loc[:, 'elevation_angle'] = elevation_angle
print(distances, azimuths, elevation_angle)

# Assuming you have a list of feature columns and target columns
feature_columns = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'pressure', 'temperature', 'humidity', 'gas_resistance']  # Adjust as per your data
target_columns = ['rsrp', 'rsrq', 'rssi', 'sinr']

# Separate features and targets from the new dataset
X_new = new_data[feature_columns]
y_true = new_data[target_columns]

# Preprocess data for both PyTorch and Keras models
X_new_tensor, scaler = preprocess_new_data(X_new, feature_columns)

# Evaluate all models
for model_name, model in models.items():
    if model_name == 'transformer':
        mse, mae = evaluate_model(model, X_new_tensor, y_true.values, model_type='pytorch')
    else:
        mse, mae = evaluate_model(model, X_new, y_true.values, model_type='keras')
    
    print(f"{model_name} Model - MSE: {mse}, MAE: {mae}")