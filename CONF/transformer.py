import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config
from feature_engineering import compute_distance_to_tower, compute_tower_direction, compute_sinr_weighted_rssi

# class GaussianTransformer(nn.Module):
#     def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
#         super(GaussianTransformer, self).__init__()
        
#         # Ensure input_dim matches what is in train_random_features
#         print(f"Expected input_dim: {input_dim}")  # Debugging print

#         self.pos_encoding = nn.Parameter(torch.zeros(1, input_dim, embed_dim)) # pose encoding
        
#         self.embedding = nn.Linear(input_dim, embed_dim)  # Correct input dim
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.decoder = nn.Linear(embed_dim, 1)  # Predict signal strength

#     def forward(self, x):
#         x = self.embedding(x)  # Ensure embedding works with input size
#         return self.decoder(x)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class GaussianTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(GaussianTransformer, self).__init__()

        print(f"Expected input_dim: {input_dim}")  # Debugging print

        self.embedding = nn.Linear(input_dim, embed_dim)  # Maps input_dim to embed_dim

        # correct positional encoding shape: (1, 1, embed_dim) so it can be broadcasted
        self.pos_encoding = nn.Parameter(torch.zeros(1, embed_dim)) # add in desciption

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.decoder = nn.Linear(embed_dim, 1)  # Output single value (e.g., signal strength)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, embed_dim)

        #corect way to apply positional encoding
        x = x + self.pos_encoding  # Broadcasting to match (batch_size, embed_dim)

        x = self.transformer(x)  # Pass through Transformer Encoder
        return self.decoder(x)  # Predict signal strength




def generate_gaussian_features_old(data):
    """
    Extracts relevant Gaussian features for transformer input.
    
    Args:
        data (pd.DataFrame): Gaussian-processed data.
        
    Returns:
        np.array: Feature array ready for transformer training.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a DataFrame, but got {}".format(type(data)))

    # print("FOR generate_gaussian_features _>>>>>>>Available columns in data:", data.columns)
    
    required_columns = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.x', 'localPosition.y', 'localPosition.z', 'rsrp', 'rsrq', 'rssi', 'sinr', 'covariance']
    print(data.columns)
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")
    
    # Extract the features
    lat_lon_alt = data[["gps.lat", "gps.lon", "altitudeAMSL"]].values
    signal_strength = data[["rsrp", "rsrq", "rssi", "sinr"]].values
    covariance = np.array([cov.flatten() for cov in data["covariance"].values])

    # Combine features
    features = np.hstack([lat_lon_alt, signal_strength, covariance])
    
    return features

def generate_gaussian_features(data, tower_location=(52.60818, 1.542818, 15.2)):
    """
    Extracts Gaussian features, ensuring covariance is handled correctly,
    and incorporates engineered features.

    Args:
        data (pd.DataFrame): Gaussian-processed data.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the nearest cell tower.

    Returns:
        np.array: Feature array for transformer training.
    """

    # --- Standard feature columns ---
    feature_columns = ['gps.lat', 'gps.lon', 'altitudeAMSL', 
                       'localPosition.x', 'localPosition.y', 'localPosition.z', 
                       'rsrp', 'rsrq', 'rssi', 'sinr']
    
    spatial_signal_features = data[feature_columns].values

    # --- Compute new engineered features ---
    dist_to_tower = compute_distance_to_tower(data, tower_location).values.reshape(-1, 1)
    azimuth, elevation = compute_tower_direction(data, tower_location)
    sinr_weighted_rssi = compute_sinr_weighted_rssi(data).values.reshape(-1, 1)

    azimuth = azimuth.values.reshape(-1, 1)  # Ensure proper shape
    elevation = elevation.values.reshape(-1, 1)

    # --- Ensure covariance is extracted correctly ---
    if "covariance" in data.columns:
        covariance_features = np.array([cov.flatten() for cov in data["covariance"].values])
    else:
        print("WARNING: Covariance missing, using only standard features.")
        covariance_features = np.zeros((data.shape[0], 10))  # Placeholder if covariance is missing

    scaler = StandardScaler()
    covariance_features = scaler.fit_transform(covariance_features)

    # --- Combine all features ---
    features = np.hstack([
        spatial_signal_features,  # Standard features
        sinr_weighted_rssi,       # New feature
        dist_to_tower,            # New feature
        azimuth, elevation,       # New directional features
        covariance_features       # Covariance matrix
    ])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)


    # Debugging output to verify structure
    print(f"Final Shape of Gaussian Features: {features.shape}")
    print(f"Sample Feature Row:\n {features[0]}")
    print("Feature means:", np.mean(features, axis=0))
    print("Feature stds:", np.std(features, axis=0))
    print("Min values:", np.min(features, axis=0))
    print("Max values:", np.max(features, axis=0))
    print("Any NaNs?", np.isnan(features).any())

    return features


def train_gaussian_transformer(features, train_data, epochs=20, lr=0.001):
    X_train = torch.tensor(features, dtype=torch.float32)

    y_train = torch.tensor(train_data["rssi"].values, dtype=torch.float32).unsqueeze(1)

    print(f"Final shape of training features: {X_train}")  # Debugging print

    # Fix the transformer input dimension
    model = GaussianTransformer(input_dim=X_train.shape[1])  

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # sqr is diffrentiable

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    return model





def evaluate_gaussian_transformer(model, test_data, test_features):
    """
    Evaluates the Gaussian Transformer model.

    Args:
        model (nn.Module): Trained Transformer model.
        test_data (pd.DataFrame): Test dataset.
        test_features (np.array): Gaussian-transformed test features.

    Returns:
        float: Mean Absolute Error (MAE).
        tensor: y_pred - beware it will mess up other funcs, remove if homogenious errors arise
    """
    X_test = torch.tensor(test_features, dtype=torch.float32)
    y_test = torch.tensor(test_data["rssi"].values, dtype=torch.float32).unsqueeze(1)
    
    # Ensure correct shape
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Mismatch: X_test has {X_test.shape[0]} samples, but y_test has {y_test.shape[0]} samples.")

    # Model evaluation
    with torch.no_grad():
        y_pred = model(X_test)

    # Compute metrics
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()

    return mae
    # return mae, rmse



