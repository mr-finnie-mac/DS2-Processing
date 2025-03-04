import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd

class GaussianTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(GaussianTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, 1)  # Predict signal strength

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.decoder(x)
    

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

    print("FOR generate_gaussian_features _>>>>>>>Available columns in data:", data.columns)
    
    required_columns = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.x', 'localPosition.y', 'localPosition.z', 'rsrp', 'rsrq', 'rssi', 'sinr', 'covariance']
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

from sklearn.preprocessing import StandardScaler

def generate_gaussian_features(data):
    """Ensure input data is a DataFrame before feature scaling."""

    # Convert to DataFrame if necessary
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)  # Convert list/array to DataFrame

    feature_cols = data.columns  # Get column names dynamically

    # Normalize transformer input
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])  

    return data.values  # Return NumPy array for transformer model



def train_gaussian_transformer(features, train_data, test_data, epochs=20, lr=0.001):
    # Ensure input is a DataFrame
    if isinstance(train_data, np.ndarray):
        train_data = pd.DataFrame(train_data)

    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("Expected a DataFrame but got NumPy array. Check data flow.")

    print("Type of train data:", type(train_data))  # Debugging check

    # Generate Gaussian features
    # gaussians = generate_gaussian_features(train_data)
    gaussians = features

    # Verify shape of generated features
    print("Shape of generated Gaussian features:", [g.shape for g in gaussians], len(gaussians))
    print("Compsition of a gaussain feature: ", gaussians[10])

    # Create input tensor for training
    X_train = torch.tensor([np.hstack([g[0], g[2]]) for g in gaussians], dtype=torch.float32)

    # Ensure the correct shape for y_train
    y_train = torch.tensor(train_data["rssi"].values, dtype=torch.float32).unsqueeze(1)

    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    model = GaussianTransformer(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    return model








def evaluate_gaussian_transformer(model, test_data, gaussians_test):
    # Generate test features from the Gaussian features
    # X_test = generate_gaussian_features(test_data)  # Uncomment if needed
    
    # Create input tensor from Gaussian features
    X_test = torch.tensor([np.hstack([g[0], g[2]]) for g in gaussians_test], dtype=torch.float32)
    
    # Extracting the true labels for RSSI
    y_test = torch.tensor(test_data["rssi"].values, dtype=torch.float32).unsqueeze(1)
    
    # Check shapes of X_test and y_test
    print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")
    
    # Ensure that the number of samples matches
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X_test has {X_test.shape[0]} samples, "
                         f"but y_test has {y_test.shape[0]} samples.")
    
    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X_test)

    # Calculate MAE and RMSE
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
    
    # Return both metrics
    # return  mae, "RMSE": rmse}
    return mae


