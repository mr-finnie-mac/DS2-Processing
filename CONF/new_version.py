from config import *
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from IDW import perform_IDW
from krig import perform_Krig
from ensemble import perform_ensemble
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from spatial_test_train_split import perform_splits
from unet import preprocess_unet_data, build_unet, plot_unet_heatmap
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.losses import Huber
import sys
from transformer import train_gaussian_transformer, evaluate_gaussian_transformer, generate_gaussian_features
from gaussian import compute_anisotropic_covariance, plot_clusters_with_gaussians, cluster_and_assign_means, create_gaussians_for_clusters, adaptively_cluster_points, new_generate_gaussian_features, create_gaussian_representation, plot_gaussian_splats
from feature_engineering import compute_tower_direction_local, compute_distance_to_tower, compute_tower_direction, compute_sinr_weighted_rssi, plot_flightpath_with_arrows, plot_flightpath_with_distances, plot_sinr_weighted_rssi, plot_flightpath_with_diamonds, plot_flightpath_combined
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
import torch.nn as nn
from eval_baselines import *
# import torch.optim as optim
import joblib
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Positional Encoding Layer (Now Global)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Define Transformer Model (Now Global)
class SignalQualityTransformer(nn.Module):
    def __init__(self, input_dim, d_model=264, nhead=4, num_layers=2):
        super(SignalQualityTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Embed features
        self.pos_encoder = PositionalEncoding(d_model)  # Add position info
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 3)  # Output RSRP, RSRQ, SINR predictions

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, seq_len=1, d_model)
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer(x)  # Self-attention
        x = self.fc(x.squeeze(1))  # (batch_size, 3)
        return x

def new_method(file_name="placeholder", input_df = [], model_name="x"):
    plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 12,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.linewidth": 3,
            "lines.markersize": 10
        })

    # Input
    # Load dataset
    dataset = input_df
    if file_name != "placeholder":
        dataset = pd.read_csv(file_name)


    train = dataset[['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.x','localPosition.y','localPosition.z', 'pressure', 'temperature', 'humidity', 'gas_resistance']]
    train = train.copy()  # Ensure it's a separate DataFrame before modifying
    
    # Feature engineering
    # Tower distance
    distances = compute_distance_to_tower(train, tower_location=tower_position)/1000
    # train['distance_to_tower'] = distances
    train.loc[:, 'distance_to_tower'] = distances
    # Tower direction
    _, elevation_angle = compute_tower_direction_local(train)
    azimuths,_ = compute_tower_direction(train, tower_location=tower_position)
    # train['azimuth'] = azimuths # add az to input
    train.loc[:, 'azimuth'] = azimuths
    # train['elevation_angle'] = elevation_angle # add e angle to input
    train.loc[:, 'elevation_angle'] = elevation_angle
    print(distances, azimuths, elevation_angle)


    # Input features: spatial relationships + absolute positioning
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'pressure', 'temperature', 'humidity', 'gas_resistance']
    if train.isna().sum().sum() > 0:
        print("Warning: Missing values detected in training data! Filling with mean...")
        train.fillna(train.mean(), inplace=True)  # Handle missing data

    print(train[features])
    train[features].to_csv("testing2.csv")
    # for testing
    sinr_values = input_df['sinr'].values  # SINR for loss weighting
    true_rsrp = input_df['rsrp'].values  # Actual RSRP values
    true_rsrq = input_df['rsrq'].values  # Actual RSRQ values
    true_sinr = input_df['sinr'].values  # Actual SINR values

    # Normalize input features
    scaler = StandardScaler()
    X = scaler.fit_transform(train[features].values)

    joblib.dump(scaler, f"{model_name}_scaler.pkl")  # ssave the actual StandardScaler instance

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.stack([
        torch.tensor(true_rsrp, dtype=torch.float32),
        torch.tensor(true_rsrq, dtype=torch.float32),
        torch.tensor(true_sinr, dtype=torch.float32)
    ], dim=1)  # Stack all three as a 2D tensor

    # Positional Encoding Layer
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super(PositionalEncoding, self).__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :].to(x.device)
        
    # Define Transformer Model
    class SignalQualityTransformer(nn.Module):
        def __init__(self, input_dim, d_model=264, nhead=4, num_layers=2):
            super(SignalQualityTransformer, self).__init__()
            self.embedding = nn.Linear(input_dim, d_model)  # Embed features
            self.pos_encoder = PositionalEncoding(d_model)  # Add position info
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
                num_layers=num_layers
            )
            self.fc = nn.Linear(d_model, 3)  # Output RSRP, RSRQ, SINR predictions

        def forward(self, x):
            x = self.embedding(x).unsqueeze(1)  # (batch_size, seq_len=1, d_model)
            x = self.pos_encoder(x)  # Add positional encoding
            x = self.transformer(x)  # Self-attention
            x = self.fc(x.squeeze(1))  # (batch_size, 3)
            return x

    # Initialize Model
    print(f"X TENSOR{len(features)}")
    model = SignalQualityTransformer(input_dim=len(features))

    # SINR-weighted Loss Function
    class WeightedMSELoss(nn.Module):
        def __init__(self, sinr_values):
            super(WeightedMSELoss, self).__init__()
            # Create the weights tensor by expanding it to match the shape of the targets (batch_size, 3)
            self.weights = torch.exp(-torch.tensor(sinr_values, dtype=torch.float32)).unsqueeze(1)  # (batch_size, 1)
            self.weights = self.weights.expand(-1, 3)  # Now (batch_size, 3) to match the prediction/target shape

        def forward(self, predictions, targets):
            # Calculate weighted MSE
            loss = self.weights * (predictions - targets) ** 2
            return loss.mean()  # Averaging over all elements

    # Define loss and optimizer
    # criterion = WeightedMSELoss(sinr_values)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            # During training, save the fitted scaler
    
    


    torch.save(model.state_dict(), f"{model_name}_transformer_model.pth")


 # Show results as figure
    latitudes = dataset['gps.lat'].values
    longitudes = dataset['gps.lon'].values

    # Get model predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predicted_signal_quality = model(X_tensor).numpy()

    # Separate predicted values for RSRP, RSRQ, and SINR
    predicted_rsrp, predicted_rsrq, predicted_sinr = predicted_signal_quality[:, 0], predicted_signal_quality[:, 1], predicted_signal_quality[:, 2]

    # Create side-by-side scatter plots (6 in total)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)

    # Real vs Predicted Plots for RSRP, RSRQ, SINR
    metrics = ["RSRP", "RSRQ", "SINR"]
    true_values = [input_df['rsrp'].values, input_df['rsrq'].values, input_df['sinr'].values]
    predicted_values = [predicted_rsrp, predicted_rsrq, predicted_sinr]

    for i, metric in enumerate(metrics):
        # True data
        sc1 = axes[0, i].scatter(longitudes, latitudes, c=true_values[i], cmap='coolwarm', edgecolor='k', alpha=0.75)
        axes[0, i].set_title(f"True {metric} Values")
        fig.colorbar(sc1, ax=axes[0, i], label=metric)

        # Predicted data
        sc2 = axes[1, i].scatter(longitudes, latitudes, c=predicted_values[i], cmap='coolwarm', edgecolor='k', alpha=0.75)
        axes[1, i].set_title(f"Predicted {metric} Values")
        fig.colorbar(sc2, ax=axes[1, i], label=metric)

    # Labels
    for ax in axes.flat:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    if show_res:
        plt.show()

    # Save output to CSV with both true and predicted values
    output_df = pd.DataFrame({
        'predicted_rsrp': predicted_rsrp,
        'predicted_rsrq': predicted_rsrq,
        'predicted_sinr': predicted_sinr,
        'true_rsrp': true_values[0],
        'true_rsrq': true_values[1],
        'true_sinr': true_values[2]
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"{file_name}_{timestamp}.csv"
    output_df.to_csv(filename)

    return output_df

import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test_split(df, test_size):
    """
    Create a train-test split based on the given test size
    test_size: fraction of the dataset to use for testing eg 0.1 for 10% test data
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

def configure_for_transformer(train_df, test_df, placeholder_value=-999):
    """
    Prepare dataset for transformer by replacing test set signal values and merging train+test.

    Args:
        train_df (DataFrame): The training dataset.
        test_df (DataFrame): The test dataset.
        placeholder_value (float): Value to replace test signal readings.

    Returns:
        transformer_df (DataFrame): The combined dataset with test signal values removed.
    """
    signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]  # Define signal columns

    # Copy test set and remove signal readings
    test_df_no_signals = test_df.copy()
    test_df_no_signals[signal_cols] = placeholder_value  # Mask test signal readings

    # Combine train and modified test set
    transformer_df = pd.concat([train_df, test_df_no_signals], axis=0)

    return transformer_df


def create_train_test_splits_helpers(df, test_size=0.2, position_cols=None, signal_cols=None):
    """
    Create two train-test splits:
    1. 'helpers': Train set has some signal readings, test set has none.
    2. 'pure_spatial': Neither train nor test has signal readings.
    
    Args:
        df: Original dataset (pandas DataFrame)
        test_size: Fraction of data to use for testing
        position_cols: List of column names representing spatial positions
        signal_cols: List of column names representing signal readings
    
    Returns:
        Tuple (train_helpers, test_helpers, train_pure_spatial, test_pure_spatial)
    """
    if position_cols is None:
        position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    if signal_cols is None:
        signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]
    
    # Split dataset into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Create 'helpers' version (train set keeps signal readings, test removes them)
    train_helpers = train_df.copy()
    test_helpers = test_df.copy()
    test_helpers[signal_cols] = np.nan  # Remove signal readings from test
    
    # Create 'pure_spatial' version (remove signal readings from both train & test)
    train_pure_spatial = train_df.copy()
    test_pure_spatial = test_df.copy()
    train_pure_spatial[signal_cols] = np.nan  # Remove from train as well
    test_pure_spatial[signal_cols] = np.nan  # Remove from test
    
    return train_helpers, test_helpers, train_pure_spatial, test_pure_spatial

def create_transformer_splits(df, test_size=0.2, helpers_percentage=1.0):
    """
    Creates train/test splits and prepares datasets for the transformer method.
    
    Args:
        df (DataFrame): The full dataset.
        test_size (float): The proportion of data to use for testing.
        helpers_percentage (float): Fraction of train set signal readings to keep (0-1).

    Returns:
        train_pure_spatial (DataFrame): Train set with only spatial data.
        train_helpers (DataFrame): Train set with spatial + limited signal readings.
        test_set (DataFrame): Test set with spatial data only.
    """
    # Define signal and position columns
    signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]  # Adjust based on dataset
    spatial_cols = [col for col in df.columns if col not in signal_cols]  # Everything else
    
    # Perform the initial train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_pure_spatial = train_df[spatial_cols].copy() + test_df[spatial_cols].copy() # this should be the spatial data fro all dataset + the signal readings for train set
    train_helpers = train_df + test_df[spatial_cols].copy() # this should be the spatial data fro all dataset + the signal readings for train set
    
    # Create the two versions of the training set
    train_pure_spatial = train_df[spatial_cols].copy()  # No signal readings at all
    train_helpers = train_df.copy()  # Contains all spatial + signal readings
    
    # If helpers_percentage < 1.0, randomly remove some signal readings from train_helpers
    if helpers_percentage < 1.0:
        for col in signal_cols:
            mask = np.random.rand(len(train_helpers)) > helpers_percentage  # Mask to drop rows
            train_helpers.loc[mask, col] = np.nan  # Set some signal readings to NaN

    return train_df, test_df, train_pure_spatial, train_helpers

def evaluate_baselines(df, test_sizes):
    error_data_random = []
    error_data_block = []
    
    # Iterate through each test size from 10% to 50% downsampling
    for test_size in test_sizes:
        # Create train and test split
        train_df, test_df = create_train_test_split(df, test_size)

        # IDW
        rand_mae, _, block_mae, _ = do_IDW(test_size, train_df, test_df, train_df, test_df)
        error_data_random.append(["IDW", test_size, rand_mae])
        error_data_block.append(["IDW", test_size, block_mae])

        # Kriging
        rand_mae, _, block_mae, _ = do_Krig(test_size, train_df, test_df, train_df, test_df)
        error_data_random.append(["Kriging", test_size, rand_mae])
        error_data_block.append(["Kriging", test_size, block_mae])

        # Ensemble Models
        rf_rand_mae, _, rf_block_mae, _, \
        xg_rand_mae, _, xg_block_mae, _, \
        lg_rand_mae, _, lg_block_mae, _, \
        ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(test_size, train_df, test_df, train_df, test_df)

        print(len(train_df))
        train_df.to_csv("testing.csv")
        # transformer
        trans_ouput = new_method(input_df=train_df)
        mae_rsrp = mean_absolute_error(trans_ouput["true_rsrp"], trans_ouput["predicted_rsrp"])
        # mae_rsrq = mean_absolute_error(transformer_results["true_rsrq"], transformer_results["predicted_rsrq"])
        # mae_sinr = mean_absolute_error(transformer_results["true_sinr"], transformer_results["predicted_sinr"])
    

        
        
        error_data_random.append(["Ensemble", test_size, ens_rand_mae])
        error_data_random.append(["Transformer", test_size, mae_rsrp])
        # error_data_block.append(["Ensemble", test_size, ens_block_mae])

    return error_data_random, error_data_block

def compare_all_methods(file_name="CONF/cleaned_rows.csv"):
    # Load dataset
    df = pd.read_csv(file_name)
    
    # Define test sizes (from 10% to 50%)
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Evaluate Baseline Models (IDW, Kriging, Ensemble)
    base_mae,_ = evaluate_baselines(df, test_sizes)

    # Evaluate Transformer Model
    # transformer_results_random, transformer_results_block = evaluate_transformer(df, test_sizes)

    # Combine results for comparison
    # all_results_random = error_data_random + transformer_results_random
    # all_results_block = error_data_block + transformer_results_block

    # Convert results to DataFrame
    base_mae = pd.DataFrame(base_mae, columns=["Method", "Test Size", "MAE"])
    # block_results_df = pd.DataFrame(all_results_block, columns=["Method", "Test Size", "MAE"])
    print(base_mae)

    # Save the results to CSV for later comparison
    base_mae.to_csv('comparison_random_results.csv', index=False)
    # block_results_df.to_csv('comparison_block_results.csv', index=False)

    print("Comparison results saved!")
    return base_mae

# visualise

    
def plot_results(df, title):
        """
        Line plot with distinct symbols and line styles for single-layer prediction results.

        Args:
            df: DataFrame containing results (Method, Test Size, MAE)
            title: Title for the plot
        """

        # --- Global Font & Style Settings ---
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 12,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.linewidth": 3,
            "lines.markersize": 10
        })

        # Ensure correct data types
        df['Method'] = df['Method'].astype('category')
        df['Test Size'] = df['Test Size'].astype(float)
        df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")  # Convert to float

        fig, ax = plt.subplots(figsize=(9, 6))  # Adjusted for readability in small figures

        # --- Line Plot with Symbols and Custom Line Styles ---
        markers = {
            "IDW": "o", "Kriging": "s", "Ensemble": "D",
            "U-Net": "^", "Transformer": "X"
        }
        linestyles = {
            "IDW": "--", "Kriging": "-.", "Ensemble": ":", 
            "U-Net": (0, (3, 1, 1, 1)), "Transformer": "-"  # Splatformer always solid
        }

        unique_methods = df["Method"].unique()
        for method in unique_methods:
            method_df = df[df["Method"] == method]
            sns.lineplot(
                data=method_df, x="Test Size", y="MAE",
                label=method, marker=markers.get(method, "o"),  
                linestyle=linestyles.get(method, "--"),  
                markersize=10, linewidth=3, ax=ax  # **Thicker Lines & Larger Markers**
            )

        ax.set_title(f"{title} - Line Plot", fontsize=20)
        ax.set_xlabel("Test Size", fontsize=18)
        ax.set_ylabel("MAE", fontsize=18)
        ax.grid(True, linestyle="dotted", linewidth=1.5)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()

        # --- Save figure as PDF ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        filename = f"fig_performance_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved as {filename}")

        plt.show()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def test_cross_pretrained(file_name="CONF/cleaned_spiral.csv", test_size = 0.5, random_state = 32, RBs = 63):
    og_dataset = pd.read_csv("CONF/cleaned_rows.csv")
    new_dataset = pd.read_csv(file_name)  
    rsrps, rsrqs, sinrs, rssis = [], [], [], []  
    test_size = test_size
    
    # og_dataset_train_df, bs_test_df = train_test_split(og_dataset, test_size=test_size, random_state=random_state)
    new_dataset_train_df, new_dataset_test_df = train_test_split(new_dataset, test_size=test_size, random_state=random_state)

    # IDW
    for target, results_list in zip(["rsrp", "rsrq", "sinr"], [rsrps, rsrqs, sinrs]):
        rand_mae, _, _, _ = do_IDW(size=test_size, target=target, this_train_random=og_dataset, this_test_random=new_dataset_train_df,
                                   this_train_block=og_dataset, this_test_block=new_dataset_train_df)
        results_list.append(["IDW", test_size, rand_mae])

    # Kriging
    for target, results_list in zip(["rsrp", "rsrq", "sinr"], [rsrps, rsrqs, sinrs]):
        rand_mae, _, _, _ = do_Krig(size=test_size, target=target, this_train_random=og_dataset, this_test_random=new_dataset_train_df,
                                    this_train_block=og_dataset, this_test_block=new_dataset_train_df)
        results_list.append(["Krig", test_size, rand_mae])

    # Ensemble Models
    for target, results_list in zip(["rsrp", "rsrq", "sinr"], [rsrps, rsrqs, sinrs]):
        rf_rand_mae, _, rf_block_mae, _, \
        xg_rand_mae, _, xg_block_mae, _, \
        lg_rand_mae, _, lg_block_mae, _, \
        ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(target=target, size=test_size,
                                                        this_train_random=og_dataset, this_test_random=new_dataset_train_df,
                                                        this_train_block=og_dataset, this_test_block=new_dataset_train_df)
        results_list.append(["Ensemble", test_size, ens_rand_mae])

    # Load Transformer Model
    transformer_model = SignalQualityTransformer(input_dim=10)  # 10d input size
    transformer_model.load_state_dict(torch.load("perim_transformer_model.pth", weights_only=True))

    transformer_model.eval()  # Set to eval mode

    # Feature Engineering
    distances = compute_distance_to_tower(new_dataset_train_df, tower_location=tower_position) / 1000
    new_dataset_train_df.loc[:, 'distance_to_tower'] = distances
    _, elevation_angle = compute_tower_direction_local(new_dataset_train_df)
    azimuths, _ = compute_tower_direction(new_dataset_train_df, tower_location=tower_position)
    new_dataset_train_df.loc[:, 'azimuth'] = azimuths
    new_dataset_train_df.loc[:, 'elevation_angle'] = elevation_angle

    # Select features (MUST match training setup)
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 
                'gps.lat', 'gps.lon', 'altitudeAMSL', 
                'pressure', 'temperature', 'humidity', 'gas_resistance']

    X_test = new_dataset_train_df[features].values

    # Load the pre-trained scaler
    scaler = joblib.load("perim_scaler.pkl")  # Load the scaler that was saved during training

    assert isinstance(scaler, StandardScaler), "Loaded scaler is not a StandardScaler object!"
    X_test_scaled = scaler.transform(X_test)  
    # Transform the test data (DO NOT fit again!)
    X_test_scaled = scaler.transform(X_test)  

    # Convert to PyTorch tensor & add sequence dimension for Transformer
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


    predictions = {}

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.to(device)  # Move model to device
    X_test_tensor = X_test_tensor.to(device)  # Move data to same device

    # Transformer Model Predictions
    with torch.no_grad():
        transformer_preds = transformer_model(X_test_tensor).cpu().numpy()

    # Convert predictions to NumPy & remove NaNs
    # transformer_preds = transformer_preds.cpu().numpy()  # Move to CPU & convert to NumPy
    transformer_preds = np.nan_to_num(transformer_preds, nan=0.0)  # Replace NaNs with 0

    # Store in dictionary
    predictions["Transformer"] = transformer_preds

    # Convert to DataFrame
    pred_df = pd.DataFrame(transformer_preds, columns=["rsrp_pred", "rsrq_pred", "sinr_pred"])
    pred_df["method"] = "Transformer" 

    # Compute Mean Absolute Error
    y_true = new_dataset_train_df[['rsrp', 'rsrq', 'sinr']].values

    for model_name, y_pred in predictions.items():
        # Ensure y_pred is a NumPy array & remove NaNs
        y_pred = np.nan_to_num(np.array(y_pred), nan=0.0)
        
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{model_name} MAE: {mae:.3f}")

    # Append Transformer Results to Metrics
    rsrps.append(["Transformer", test_size, mean_absolute_error(y_true[:, 0], transformer_preds[:, 0])])
    rsrqs.append(["Transformer", test_size, mean_absolute_error(y_true[:, 1], transformer_preds[:, 1])])
    sinrs.append(["Transformer", test_size, mean_absolute_error(y_true[:, 2], transformer_preds[:, 2])])

    # Compute RSSI for all models
    def compute_rssi(rsrp, rsrq, RBs=RBs):
        return rsrp + 10 * np.log10(RBs) - rsrq  # Derived formula

    for method, _, pred in rsrps:
        if method != "Transformer":
            rssi_pred = compute_rssi(pred, rsrqs[-1][2])
            rssis.append([method, 1.0 - test_size, rssi_pred])

    # Compute RSSI for Transformer predictions
    rssi_transformer = compute_rssi(transformer_preds[:, 0], transformer_preds[:, 1])
    rssis.append(["Transformer", test_size, np.mean(rssi_transformer)])

    # Create DataFrame to store all results
    results_df = pd.DataFrame(rsrps + rsrqs + sinrs + rssis, columns=["Method", "Test_Size", "MAE"])
    
    # Save results
    results_df.to_csv("cross_ds_unseen.csv", index=False)

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for method in results_df["Method"].unique():
    #     subset = results_df[results_df["Method"] == method]
    #     plt.plot(subset["Test_Size"], subset["MAE"], marker="o", label=method)

    # plt.xlabel("Test Size")
    # plt.ylabel("Mean Absolute Error (MAE)")
    # plt.title("Model Performance on Unseen Dataset")
    # plt.legend()
    # plt.grid()
    # plt.show()

    print("Results saved to cross_ds_unseen.csv")
    return results_df


def add_features(dataset):
    # Feature Engineering
    distances = compute_distance_to_tower(dataset, tower_location=tower_position) / 1000
    dataset.loc[:, 'distance_to_tower'] = distances
    _, elevation_angle = compute_tower_direction_local(dataset)
    azimuths, _ = compute_tower_direction(dataset, tower_location=tower_position)
    dataset.loc[:, 'azimuth'] = azimuths
    dataset.loc[:, 'elevation_angle'] = elevation_angle

    
    return dataset


def plot_cross_all_props(file_path="per_prop.xlsx"):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Define signal properties and corresponding columns
    signals = {
        "RSRP": ["RSRP Method", "RSRP Test_Size", "RSRP MAE"],
        "RSRQ": ["RSRQ Method", "RSRQ Test_Size", "RSRQ MAE"],
        "SINR": ["SINR Method", "SINR Test_Size", "SINR MAE"]
    }

    # Define markers and line styles for different methods
    line_styles = {
        "Ensemble": ("o", "-"),  # Circle marker, solid line
        "IDW": ("s", "--"),  # Square marker, dashed line
        "Krig": ("^", "-."),  # Triangle marker, dash-dot line
        "Transformer": ("d", ":")  # Diamond marker, dotted line
    }

    # Apply font and style settings
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 3,
        "lines.markersize": 10
    })

    for i, (signal, cols) in enumerate(signals.items(), 1):
        plt.subplot(3, 1, i)  # Create a subplot for each signal
        for method in df[cols[0]].unique():  # Unique methods (Ensemble, IDW, etc.)
            subset = df[df[cols[0]] == method]
            marker, linestyle = line_styles.get(method, ("o", "-"))  # Default if method not in dict
            plt.plot(subset[cols[1]], subset[cols[2]], label=method, marker=marker, linestyle=linestyle)

        plt.grid(True)

        # Only show labels on the last subplot
        if i == 3:
            plt.xlabel("Test Size")
            plt.ylabel("MAE")
        else:
            plt.xlabel("")
            plt.ylabel("")

    # Add legend only once (outside the plots)
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 3.2), ncol=1)
    
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# # Function to run the comparisons and collect results
# def run_comparisons(dataset, cross_dataset, feature_list, target_column, random_states=[42, 7, 11, 22, 33, 44, 55, 66, 77, 88]):
#     comparison_rsrp = []  # To store comparison results
#     error_data_block = []  # To store block error data
    
#     # Initialize averages for each model
#     avg_results = {
#         'MLP': {'mae': [], 'rmse': []},
#         'IDW': {'mae': [], 'rmse': []},
#         'Kriging': {'mae': [], 'rmse': []},
#         'Ensemble': {'mae': [], 'rmse': []}
#     }
#     cross_dataset
#     # Split the datasets first
#     X_cross_test = cross_dataset[feature_list].values
#     Y_cross_test = cross_dataset[target_column].values
    
#     # Loop over different random states for repeated training and evaluation
#     for rand_state in random_states:
#         print(f"Running with random state: {rand_state}")
#         test_size = 0.8

#         # Split dataset into train and test sets
#         train_x, test_set = train_test_split(dataset, test_size=test_size, random_state=rand_state)
   
#         # Features and target variable
#         X_train = train_x[feature_list].values
#         y_train = train_x[target_column].values
#         X_test = test_set[feature_list].values
#         y_test = test_set[target_column].values

#         # --- MLP Model ---
#         mlp_model = run_mlp(input_dim=len(feature_list), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=1000)
        
#         test_mae, test_rmse = evaluate_mlp(model=mlp_model, X_test=X_test, y_test=y_test)
#         cross_mae, cross_rmse = evaluate_mlp(model=mlp_model, X_test=X_cross_test, y_test=Y_cross_test)
        
#         # avg_results['MLP']['mae'].append(test_mae)
#         # avg_results['MLP']['rmse'].append(test_rmse)
#         avg_results['MLP']['mae'].append(cross_mae)
#         avg_results['MLP']['rmse'].append(cross_rmse)
        
#         print(f"MLP - Test Set (Same Path): MAE {test_mae}, RMSE {test_rmse}")
#         print(f"MLP - Cross-Test Set (Different Path): MAE {cross_mae}, RMSE {cross_rmse}")
#         cross_dataset = cross_dataset[~cross_dataset.index.isin(train_x.index)]
#         common_points = cross_dataset[cross_dataset.index.isin(train_x.index)]
        
        
#         print(f"Number of test points also in train set: {len(common_points)} size of stipper {len(cross_dataset['gps.lat'].values)}")
#         # --- IDW Model ---
#         rand_mae, rand_rmse, block_mae, _ = do_IDW(target = "rssi", size=test_size, this_train_random=train_x, this_test_random=cross_dataset, this_train_block=train_x, this_test_block=test_set)
#         comparison_rsrp.append(["IDW", test_size, rand_mae])
#         error_data_block.append(["IDW", test_size, rand_rmse])
        
#         avg_results['IDW']['mae'].append(rand_mae)
#         avg_results['IDW']['rmse'].append(rand_rmse)

#         # --- Kriging Model ---
#         rand_mae, rand_rmse, block_mae, _ = do_Krig(target = "rssi", size=test_size, this_train_random=train_x, this_test_random=cross_dataset, this_train_block=train_x, this_test_block=test_set)
#         comparison_rsrp.append(["Kriging", test_size, rand_mae])
#         error_data_block.append(["Kriging", test_size, rand_rmse])
        
#         avg_results['Kriging']['mae'].append(rand_mae)
#         avg_results['Kriging']['rmse'].append(rand_rmse)
        
#         # --- Ensemble Model ---
#         rf_rand_mae, _, rf_block_mae, _, \
#         xg_rand_mae, _, xg_block_mae, _, \
#         lg_rand_mae, _, lg_block_mae, _, \
#         ens_rand_mae, ens_rand_rmse, ens_block_mae, _ = do_Ensemble(target = "rssi", size=test_size, this_train_random=train_x, this_test_random=cross_dataset, this_train_block=train_x, this_test_block=test_set)
        
#         comparison_rsrp.append(["Ensemble", test_size, ens_rand_mae])
#         error_data_block.append(["Ensemble", test_size, ens_rand_rmse])
        
#         avg_results['Ensemble']['mae'].append(ens_rand_mae)
#         avg_results['Ensemble']['rmse'].append(ens_block_mae)

#     # After looping through all random states, calculate averages for each model
#     for model, results in avg_results.items():
#         avg_mae = np.mean(results['mae'])
#         avg_rmse = np.mean(results['rmse'])
#         print(f"\nAverage Results for {model}:")
#         print(f"  Average MAE: {avg_mae:.4f}")
#         print(f"  Average RMSE: {avg_rmse:.4f}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_datasets(dataset1, dataset2):
    """
    Plot two datasets in 3D space (GPS Latitude, Longitude, Altitude).

    Args:
        dataset1: First dataset (plotted in RED).
        dataset2: Second dataset (plotted in GREEN).
    """

    # Apply font and style settings
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 3,
        "lines.markersize": 10
    })
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    lat1, lon1, alt1 = dataset1["gps.lat"], dataset1["gps.lon"], dataset1["altitudeAMSL"]
    lat2, lon2, alt2 = dataset2["gps.lat"], dataset2["gps.lon"], dataset2["altitudeAMSL"]

    # Plot datasets
    ax.scatter(lat1, lon1, alt1, c='red', label="ROWS", alpha=0.6)
    ax.scatter(lat2, lon2, alt2, c='green', label="SPRIRAL", alpha=0.6)

    # Labels
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_zlabel("Altitude (AMSL)")
    ax.set_title("3D Spatial Plot of Two Datasets")
    ax.legend()

    plt.show()

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
def spatial_block_split(dataset, test_size=0.2, n_clusters=10):
    """
    Perform a spatially-aware train-test split by selecting a 'block' of nearby data points.

    Args:
        dataset: DataFrame containing 'gps.lat' and 'gps.lon' columns.
        test_size: Fraction of data to use for testing.
        n_clusters: Number of spatial clusters to form.

    Returns:
        train_set: DataFrame for training.
        test_set: DataFrame for testing.
    """
    coords = dataset[['gps.lat', 'gps.lon']].values

    # Perform KMeans clustering to group spatially close points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    dataset['cluster'] = kmeans.fit_predict(coords)

    # Find the largest cluster (or a random one)
    largest_cluster = dataset['cluster'].value_counts().idxmax()  # Cluster with most points
    test_set = dataset[dataset['cluster'] == largest_cluster]  # Use one cluster as test set

    # Ensure test set size is close to the desired fraction
    if len(test_set) > len(dataset) * test_size:
        test_set = test_set.sample(frac=test_size, random_state=42)  # Subsample within cluster

    # Remaining data is the train set
    train_set = dataset.drop(test_set.index)

    # Drop the temporary cluster column
    train_set = train_set.drop(columns=['cluster'])
    test_set = test_set.drop(columns=['cluster'])

    return train_set, test_set
def temporal_block_split(dataset, timestamp_col='drone_timestamp', test_size=0.2):
    """
    Perform a temporal train-test split by sorting on the timestamp and splitting sequentially.

    Args:
        dataset: DataFrame containing the timestamp column.
        timestamp_col: Name of the column containing the timestamp.
        test_size: Fraction of data to use for testing.

    Returns:
        train_set: DataFrame for training.
        test_set: DataFrame for testing.
    """
    dataset = dataset.copy()

    # Ensure timestamp column is converted to datetime
    try:
        dataset[timestamp_col] = pd.to_datetime(dataset[timestamp_col])
    except Exception as e:
        raise ValueError(f"Failed to convert '{timestamp_col}' to datetime: {e}")

    # Sort chronologically
    dataset = dataset.sort_values(by=timestamp_col).reset_index(drop=True)

    # Split index based on test size
    split_idx = int(len(dataset) * (1 - test_size))

    # Sequential split
    train_set = dataset.iloc[:split_idx]
    test_set = dataset.iloc[split_idx:]

    return train_set, test_set


from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

def mlp_point_by_point(dataset, target_column="rssi", features=['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL'], method='ANN'):
    point_results = {
        method: {'mae': [], 'rmse': []}
    }
    csv_results = []

    dataset = add_features(dataset)
    dataset['drone_timestamp'] = pd.to_datetime(dataset['drone_timestamp'])
    dataset['timestamp_numeric'] = dataset['drone_timestamp'].astype(np.int64) // 10**9  # seconds since epoch

    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'timestamp_numeric']

    # Train/test split
    split_index = len(dataset) // 2
    train_set = dataset.iloc[:split_index]
    test_set = dataset.iloc[split_index:]
    plot_3d_datasets(train_set, test_set)


    # Scale all features including timestamp
    scaler = StandardScaler()
    scaler.fit(train_set[features].values)

    X_train = scaler.transform(train_set[features].values)
    y_train = train_set[target_column].values

    # Train the MLP
    mlp_model = run_mlp(
        input_dim=X_train.shape[1],
        X_train=X_train,
        y_train=y_train,
        X_test=X_train,  # not used, can be left out
        y_test=y_train,
        epochs=1000
    )

    # Evaluate point-by-point
    for _, test_row in test_set.iterrows():
        test_features = test_row[features].values.reshape(1, -1)
        X_test_point = scaler.transform(test_features)
        y_test_point = np.array([test_row[target_column]])

        y_pred = mlp_model.predict(X_test_point)
        mae, rmse =evaluate_mlp(model=mlp_model, X_test=X_test_point, y_test=y_test_point)

        point_results[method]['mae'].append(mae)
        point_results[method]['rmse'].append(rmse)

        csv_results.append({
            'Timestamp': test_row['drone_timestamp'],
            'lat': test_row['gps.lat'],
            'lon': test_row['gps.lon'],
            'alt': test_row['altitudeAMSL'],
            'True': y_test_point[0],
            'Predicted': y_pred[0],
            'MAE': mae,
            'RMSE': rmse
        })

    # Save results
    df = pd.DataFrame(csv_results)
    df.to_csv('method_pbyp_results.csv', index=False)
    return point_results

    


def run_comparisons(dataset, cross_dataset, feature_list, target_column, random_states=[11, 45, 35, 65, 98, 56, 28, 81, 34, 20, 26, 52, 7, 48], test_sizes=[0.1, 0.2, 0.3, 0.4, 0.5], split_type="", save_name="new-"):
    """
    Run model comparisons (MLP, IDW, Kriging, Ensemble) using consistent train/test splits.
    """
    # Initialize scalers
    scaler = StandardScaler()
    scaler.fit(dataset[feature_list].values)

    # Scaled cross-dataset for MLP
    X_cross_test = scaler.transform(cross_dataset[feature_list].values)
    Y_cross_test = cross_dataset[target_column].values

    # Store results
    avg_results = {
        'ANN': {'mae': [], 'rmse': []},
        'IDW': {'mae': [], 'rmse': []},
        'Kriging': {'mae': [], 'rmse': []},
        'Ensemble': {'mae': [], 'rmse': []},
        'LinReg': {'mae': [], 'rmse': []}
    }
    detailed_results = []

    

    for rand_state in random_states:
        for ts in test_sizes:
            print(f"Doing TS {ts}")
                
            print(f"\nRunning with random state: {rand_state}")
            test_size = ts  # Fixed test size for all models
            # Train/Test Split
            if split_type == "rand": # random
                train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=rand_state)
                cross_train_set, cross_test_set = train_test_split(cross_dataset, test_size=test_size, random_state=rand_state)
            elif split_type == "temp": # temporal
                train_set, test_set = temporal_block_split(dataset, test_size=test_size)
                cross_train_set, cross_test_set =  temporal_block_split(cross_dataset, test_size=test_size)
            elif split_type == "block": # block split
                train_set, test_set = spatial_block_split(dataset, test_size=test_size, n_clusters=4)
                cross_train_set, cross_test_set = spatial_block_split(cross_dataset, test_size=test_size, n_clusters=4)
            else:
                 train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=rand_state)
                 cross_train_set, cross_test_set = train_test_split(cross_dataset, test_size=test_size, random_state=rand_state)

            # Prepare features and target variables for training and testing
            # timestamp_features = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'drone_timestamp']
            X_train = train_set[feature_list].values
            y_train = train_set[target_column].values
            X_test = test_set[feature_list].values
            y_test = test_set[target_column].values

            # Normalize the data
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #CROSS SETUP
            X_train_cross = cross_train_set[feature_list].values
            y_train_cross = cross_train_set[target_column].values
            X_test_cross = cross_test_set[feature_list].values
            y_test_cross = cross_test_set[target_column].values

            # Normalize the data
            X_train_scaled_cross = scaler.transform(X_train_cross)
            X_test_scaled_cross = scaler.transform(X_test_cross)

            # ----- Linear Regression Baseline -----
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)

            # Evaluate on the same path test set
            y_pred_lr_test = lr_model.predict(X_test_scaled)
            test_mae_lr = median_absolute_error(y_test, y_pred_lr_test)
            test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))

            # Evaluate on the cross path test set
            y_pred_lr_cross = lr_model.predict(X_cross_test)
            cross_mae_lr = median_absolute_error(Y_cross_test, y_pred_lr_cross)
            cross_rmse_lr = np.sqrt(mean_squared_error(Y_cross_test, y_pred_lr_cross))

            # Save to avg_results
            avg_results['LinReg']['mae'].append(test_mae_lr)
            avg_results['LinReg']['rmse'].append(test_rmse_lr)

            # Save detailed results
            detailed_results.append({'model': 'LinReg', 'metric': 'mae', 'split': 'test', 'size': test_size, 'value': test_mae_lr})
            detailed_results.append({'model': 'LinReg', 'metric': 'rmse', 'split': 'test', 'size': test_size, 'value': test_rmse_lr})
            detailed_results.append({'model': 'LinReg', 'metric': 'mae', 'split': 'cross', 'size': test_size, 'value': cross_mae_lr})
            detailed_results.append({'model': 'LinReg', 'metric': 'rmse', 'split': 'cross', 'size': test_size, 'value': cross_rmse_lr})

            print(f"Linear Regression - Test Set (Same Path): MAE {test_mae_lr:.4f}, RMSE {test_rmse_lr:.4f}")
            print(f"Linear Regression - Cross-Test Set (Different Path): MAE {cross_mae_lr:.4f}, RMSE {cross_rmse_lr:.4f}")

            
            # ----- ANN -----
            mlp_model = run_mlp(input_dim=len(feature_list), X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, epochs=1000)
            mlp_model_cross = run_mlp(input_dim=len(feature_list), X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled_cross, y_test=y_test_cross, epochs=1000)
            test_mae, test_rmse = evaluate_mlp(model=mlp_model, X_test=X_test_scaled, y_test=y_test)
            cross_mae, cross_rmse = evaluate_mlp(model=mlp_model_cross, X_test=X_cross_test, y_test=Y_cross_test)

            avg_results['ANN']['mae'].append(test_mae)
            avg_results['ANN']['rmse'].append(test_rmse)

            # Save the results for ANN
            detailed_results.append({'model': 'ANN', 'metric': 'mae', 'split': 'test', 'size': test_size, 'value': test_mae})
            detailed_results.append({'model': 'ANN', 'metric': 'rmse', 'split': 'test', 'size': test_size,'value': test_rmse})
            detailed_results.append({'model': 'ANN', 'metric': 'mae', 'split': 'cross', 'size': test_size,'value': cross_mae})
            detailed_results.append({'model': 'ANN', 'metric': 'rmse', 'split': 'cross', 'size': test_size,'value': cross_rmse})

            print(f"ANN - Test Set (Same Path): MAE {test_mae:.4f}, RMSE {test_rmse:.4f}")
            print(f"MLP - Cross-Test Set (Different Path): MAE {cross_mae:.4f}, RMSE {cross_rmse:.4f}")

            # ----- IDW -----
            train_idw = train_set.copy()
            test_idw = test_set.copy()
            train_idw[feature_list] = scaler.transform(train_idw[feature_list])
            test_idw[feature_list] = scaler.transform(test_idw[feature_list])

            rand_mae, rand_rmse, block_mae, _ = do_IDW(
                target=target_column, size=test_size, 
                this_train_random=train_idw, this_test_random=test_idw,
                this_train_block=train_idw, this_test_block=test_idw, position_cols=feature_list
            )
            # Cross dataset evaluation for IDW
            cross_train_idw = cross_dataset.copy()
            cross_train_idw[feature_list] = scaler.transform(cross_train_idw[feature_list])

            cross_rand_mae, cross_rand_rmse, _, _ = do_IDW(
                target=target_column, size=test_size, 
                this_train_random=train_idw, this_test_random=cross_train_idw,
                this_train_block=train_idw, this_test_block=cross_train_idw, position_cols=feature_list
            )

            avg_results['IDW']['mae'].append(rand_mae)
            avg_results['IDW']['rmse'].append(rand_rmse)

            # Save IDW results
            detailed_results.append({'model': 'IDW', 'metric': 'mae', 'split': 'test','size': test_size, 'value': rand_mae})
            detailed_results.append({'model': 'IDW', 'metric': 'rmse', 'split': 'test','size': test_size, 'value': rand_rmse})
            detailed_results.append({'model': 'IDW', 'metric': 'mae', 'split': 'cross', 'size': test_size,'value': cross_rand_mae})
            detailed_results.append({'model': 'IDW', 'metric': 'rmse', 'split': 'cross','size': test_size, 'value': cross_rand_rmse})

            # ----- Kriging -----
            train_krig = train_set.copy()
            test_krig = test_set.copy()
            # train_krig[feature_list] = scaler.transform(train_krig[feature_list])
            # test_krig[feature_list] = scaler.transform(test_krig[feature_list])

            rand_mae, rand_rmse, block_mae, _ = do_Krig(
                target=target_column, size=test_size, 
                this_train_random=train_krig, this_test_random=test_krig,
                this_train_block=train_krig, this_test_block=test_krig, position_cols=feature_list
            )
            cross_rand_mae, cross_rand_rmse, _, _ = do_Krig(
                target=target_column, size=test_size, 
                this_train_random=train_krig, this_test_random=cross_dataset,
                this_train_block=train_krig, this_test_block=cross_dataset, position_cols=feature_list
            )

            avg_results['Kriging']['mae'].append(rand_mae)
            avg_results['Kriging']['rmse'].append(rand_rmse)

            # Save Kriging results
            detailed_results.append({'model': 'Kriging', 'metric': 'mae', 'split': 'test','size': test_size, 'value': rand_mae})
            detailed_results.append({'model': 'Kriging', 'metric': 'rmse', 'split': 'test','size': test_size, 'value': rand_rmse})
            detailed_results.append({'model': 'Kriging', 'metric': 'mae', 'split': 'cross','size': test_size, 'value': cross_rand_mae})
            detailed_results.append({'model': 'Kriging', 'metric': 'rmse', 'split': 'cross','size': test_size, 'value': cross_rand_rmse})

            # ----- Ensemble -----
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            ens_rand_mae, ens_rand_rmse, ens_block_mae, _ = do_Ensemble(
                target=target_column, size=test_size, 
                this_train_random=train_set, this_test_random=test_set,
                this_train_block=train_set, this_test_block=test_set, position_cols=feature_list
            )
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            cross_ens_rand_mae, cross_ens_rand_rmse, _, _ = do_Ensemble(
                target=target_column, size=test_size, 
                this_train_random=train_set, this_test_random=cross_dataset,
                this_train_block=train_set, this_test_block=cross_dataset, position_cols=feature_list
            )

            avg_results['Ensemble']['mae'].append(ens_rand_mae)
            avg_results['Ensemble']['rmse'].append(ens_rand_rmse)

            # Save Ensemble results
            detailed_results.append({'model': 'Ensemble', 'metric': 'mae', 'split': 'test','size': test_size, 'value': ens_rand_mae})
            detailed_results.append({'model': 'Ensemble', 'metric': 'rmse', 'split': 'test', 'size': test_size,'value': ens_rand_rmse})
            detailed_results.append({'model': 'Ensemble', 'metric': 'mae', 'split': 'cross', 'size': test_size,'value': cross_ens_rand_mae})
            detailed_results.append({'model': 'Ensemble', 'metric': 'rmse', 'split': 'cross','size': test_size, 'value': cross_ens_rand_rmse})

    # Compute Averages
    print("\nFinal Average Results:")
    for model, results in avg_results.items():
        avg_mae = np.mean(results['mae'])
        avg_rmse = np.mean(results['rmse'])
        print(f"\n{model}:")
        print(f"  - Average MAE: {avg_mae:.4f}")
        print(f"  - Average RMSE: {avg_rmse:.4f}")

    # Save detailed results to CSV
    df = pd.DataFrame(detailed_results)      
    df.to_csv(save_name+'model_comparison_results.csv', index=False)
    return df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# def plot_mae_comparison(csv1_path, csv2_path, labels=("a) Temporal Split", "b) Spiral Temporal Split"), split='test', csvFlag=True, show = True):
#     # Load CSVs
#     if csvFlag:
#         df1 = pd.read_csv(csv1_path)
#         df2 = pd.read_csv(csv2_path)
#     else:
#         df1 = csv1_path
#         df2 = csv2_path

#     # Filter for MAE only and the specified split
#     df1_mae = df1[(df1['metric'] == 'mae') & (df1['split'] == split)].copy()
#     df2_mae = df2[(df2['metric'] == 'mae') & (df2['split'] == split)].copy()

#     # Convert test size to float
#     df1_mae['size'] = df1_mae['size'].astype(float)
#     df2_mae['size'] = df2_mae['size'].astype(float)

#     # Set seaborn theme
#     sns.set(style="whitegrid")

#     # Create subplots
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#     plt.rcParams.update({
#         "font.family": "serif",
#         "font.serif": ["Times New Roman"],
#         "axes.labelsize": 18,
#         "legend.fontsize": 12,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "lines.linewidth": 3,
#         "lines.markersize": 10
#     })

#     for i, (ax, df, caption) in enumerate(zip(axes, [df1_mae, df2_mae], labels)):
#         sns.lineplot(
#             data=df,
#             x="size",
#             y="value",
#             hue="model",
#             style="model",
#             markers=True,
#             dashes=True,
#             ax=ax,
#             linewidth=plt.rcParams['lines.linewidth'],
#             markersize=plt.rcParams['lines.markersize']
#         )
#         ax.set_xlabel("Test Size")
#         if i == 0:
#             ax.set_ylabel("Median Absolute Error")
#         else:
#             ax.set_ylabel("")
#         ax.legend(title="Model")
#         if i == 0:
#             ax.get_legend().remove()  # Remove legend from first plot
        
#         # Add caption under each plot
#         ax.text(0.5, -0.25, caption, transform=ax.transAxes,
#                 ha='center', va='top', fontsize=16)

#     plt.tight_layout()
#     if show:
#         plt.show()


def plot_mae_comparison(csv1_path, csv2_path, labels=("a) Temporal Split", "b) Spiral Temporal Split"), split='test', csvFlag=True, show=True):
    # Load CSVs
    if csvFlag:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    else:
        df1 = csv1_path
        df2 = csv2_path

    # Filter for MAE and the specified split
    df1_mae = df1[(df1['metric'] == 'mae') & (df1['split'] == split)].copy()
    df2_mae = df2[(df2['metric'] == 'mae') & (df2['split'] == split)].copy()

    # Remove 'LinReg' and rename 'ANN' to 'OURS[ANN]'
    def preprocess(df):
        df = df[df['model'] != 'LinReg'].copy()
        df['model'] = df['model'].replace({'ANN': 'OURS [ANN]'})
        df['size'] = df['size'].astype(float)
        return df

    df1_mae = preprocess(df1_mae)
    df2_mae = preprocess(df2_mae)

    # Set seaborn theme
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 4,
        "lines.markersize": 12
    })

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    handles = labels_ = None  # For legend outside the plot

    for i, (ax, df, caption) in enumerate(zip(axes, [df1_mae, df2_mae], labels)):
        plot = sns.lineplot(
            data=df,
            x="size",
            y="value",
            hue="model",
            style="model",
            markers=True,
            dashes=True,
            ax=ax,
            linewidth=plt.rcParams['lines.linewidth'],
            markersize=plt.rcParams['lines.markersize']
        )

        if i == 0:
            ax.set_ylabel("Median Absolute Error")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Test Size")

        # Capture legend handles and labels from the second subplot only
        if i == 1:
            # handles, labels_ = ax.get_legend_handles_labels()
            ax.get_legend().remove()  # Remove in all plots

        # Add caption below each plot
        ax.text(0.5, -0.25, caption, transform=ax.transAxes,
                ha='center', va='top', fontsize=16)
        ax.legend(title="Model")
        if i == 0:
            ax.get_legend().remove()  # Remove legend from first plot

    # Add a single shared legend to the right of the second plot
    # if handles and labels_:
    #     fig.legend(handles, labels_, title="Model", loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, title_fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if show:
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_point_by_point_comparison(csv1_path, csv2_path, labels=("a) Method 1", "b) Method 2"), csvFlag=True):
    # Load CSVs
    if csvFlag:
        df1 = pd.read_csv(csv1_path, parse_dates=["Timestamp"])
        df2 = pd.read_csv(csv2_path, parse_dates=["Timestamp"])
    else:
        df1 = csv1_path
        df2 = csv2_path

    sns.set(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i, (ax, df, caption) in enumerate(zip(axes, [df1, df2], labels)):
        # Sort by time to make lines clean
        df = df.sort_values("Timestamp")

        # Plot True and Predicted values
        ax.plot(df["Timestamp"], df["True"], label="True RSSI", marker='o', linestyle='-', color='tab:blue')
        ax.plot(df["Timestamp"], df["Predicted"], label="Predicted RSSI", marker='x', linestyle='--', color='tab:red')

        ax.set_xlabel("Timestamp")
        if i == 0:
            ax.set_ylabel("RSSI")
        else:
            ax.set_ylabel("")

        ax.set_title(f"RSSI Comparison")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # Add subplot label
        ax.text(0.5, -0.35, caption, transform=ax.transAxes,
                ha='center', va='top', fontsize=16)

    plt.tight_layout()
    plt.show()




from keras_tuner import RandomSearch

if __name__ == '__main__':
    
    rows = "CONF/cleaned_rows.csv"
    spiral = "CONF/cleaned_spiral.csv"
    # mlp_point_by_point(dataset=pd.read_csv(spiral))

    # plot_mae_comparison(csv1_path="model_comparison_results-rows-cross-spiral.csv", csv2_path="model_comparison_results-spiral-cross-rows.csv", labels=("rows", "spiral"))

    # plot_mae_comparison(csv1_path="model_comparison_results-rows temporal.csv", csv2_path="model_comparison_results-spiral temporal.csv", labels=("rows", "spiral"))

    # Example usage:
    

    dataset = pd.read_csv(rows)
    cross_dataset = pd.read_csv(spiral)

    # Add features to datasets (e.g., distance_to_tower, azimuth, etc.)
    dataset = add_features(dataset)
    cross_dataset = add_features(cross_dataset)

    # Define features and target
    # features = ['distance_to_tower', 'azimuth', 'elevation_angle'] 
    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL', 'drone_timestamp']
    
    # PLOT STORED RESULTS
    # plot_mae_comparison(csv1_path='best_results/rows_layer_model_comparison_results.csv', csv2_path='best_results/spiral_layer_model_comparison_results.csv', labels=("a) Rows Layer (Rand)", "b) Spiral Layer (Rand)"), split='test', csvFlag=True, show=True) # present same layer plot
    # plot_mae_comparison(csv1_path='best_results/rows_layer_block_model_comparison_results.csv', csv2_path='best_results/spiral_layer_block_model_comparison_results.csv', labels=("a) Rows Layer (Block)", "b) Spiral Layer (Block)"), split='test', csvFlag=True, show=True) # present same layer plot block
    # plot_mae_comparison(csv1_path='best_results/rows_layer_model_comparison_results.csv', csv2_path='best_results/spiral_layer_model_comparison_results.csv', labels=("a) Rows Cross Layer (Rand)", "b) Spiral Cross Layer (Rand)"), split='cross', csvFlag=True, show=True) # present cross layer plot rand
    # plot_mae_comparison(csv1_path='best_results/rows_layer_block_model_comparison_results.csv', csv2_path='best_results/spiral_layer_block_model_comparison_results.csv', labels=("a) Rows Cross Layer (Block)", "b) Spiral Cross Layer (Block)"), split='cross', csvFlag=True, show=True) # present cross layer plot block
    # plot_mae_comparison(csv1_path='best_results/rows_layer_xyz_model_comparison_results.csv', csv2_path='best_results/rows_layer_aed_model_comparison_results.csv', labels=("a) Rows XYZ only (Rand)", "b) Rows AED only (Rand)"), split='test', csvFlag=True, show=True)
    # plot_mae_comparison(csv1_path='best_results/rows_layer_xyz_model_comparison_results.csv', csv2_path='best_results/rows_layer_aed_model_comparison_results.csv', labels=("a) Rows XYZ", "b) Rows AED"), split='cross', csvFlag=True, show=True)
    # plot_mae_comparison(csv1_path='best_results/rows_layer_xyz_timesplitmodel_comparison_results.csv', csv2_path='best_results/rows_layer_xyz_ts_timesplitmodel_comparison_results.csv', labels=("a) Rows XYZ", "b) Rows XYZ Timeseries"), split='test', csvFlag=True)
    # plot_mae_comparison(csv1_path='best_results/spiral_layer_xyz_model_comparison_results.csv', csv2_path='best_results/spiral_layer_aed_model_comparison_results.csv', labels=("a) Spiral Cross XYZ  (Rand)", "b) Spiral Cross AED (Rand)"), split='cross', csvFlag=True, show=True)
    plot_mae_comparison(csv1_path='best_results/spiral_layer_xyz_timesplitmodel_comparison_results.csv', csv2_path='best_results/spiral_layer_xyz_ts_timesplitmodel_comparison_results.csv', labels=("a) Spiral XYZ", "b) Spiral XYZ Timeseries"), split='test', csvFlag=True)

    # PARTIONING / SINGLE / CROSS
    target = 'rssi'
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 
                'gps.lat', 'gps.lon', 'altitudeAMSL']
    
    states = [11, 45, 35, 65, 98, 56, 28, 81, 34, 20]
    # rows_layer = run_comparisons(dataset, cross_dataset, features, target, random_states=states,split_type="rand", save_name='rows_layer_') # random same layer rows
    # print("set comp")
    # rows_layer_block = run_comparisons(dataset, cross_dataset, features, target, random_states=states, split_type="block", save_name='rows_layer_block_') # block same layer rows
    # print("set comp")
    # spiral_layer = run_comparisons(cross_dataset, dataset, features, target, random_states=states, split_type="rand", save_name='spiral_layer_') #  random same layer spiral
    # print("set comp")
    # spiral_layer_block = run_comparisons(cross_dataset, dataset, features, target, random_states=states,split_type="block", save_name='spiral_layer_block_') # block same layer spiral
    # print("set comp")
    # print("same/cross layer set comp")
    # plot_mae_comparison(csv1_path=rows_layer, csv2_path=spiral_layer, labels=("a) Rows Layer (Rand)", "b) Spiral Layer (Rand)"), split='test', csvFlag=False, show=False) # present same layer plot
    # plot_mae_comparison(csv1_path=rows_layer_block, csv2_path=spiral_layer_block, labels=("a) Rows Layer (Block)", "b) Spiral Layer (Block)"), split='test', csvFlag=False, show=False) # present same layer plot block
    # plot_mae_comparison(csv1_path=rows_layer, csv2_path=spiral_layer, labels=("a) Rows Cross Layer (Rand)", "b) Spiral Cross Layer (Rand)"), split='cross', csvFlag=False, show=False) # present cross layer plot rand
    # plot_mae_comparison(csv1_path=rows_layer_block, csv2_path=spiral_layer_block, labels=("a) Rows Cross Layer (Block)", "b) Spiral Cross Layer (Block)"), split='cross', csvFlag=False, show=True) # present cross layer plot block


    # # XYZ VS AED
    # # Run the comparison loop with random states
    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    # # # chosen_features=features
    # xyz_df = run_comparisons(dataset, cross_dataset, features, target,random_states=states, save_name='rows_layer_xyz_')
    # print("xyz done")
    # print("set comp")
    # features = ['distance_to_tower', 'azimuth', 'elevation_angle']
    # # chosen_features=features
    # aed_df = run_comparisons(dataset, cross_dataset, features, target, random_states=states, save_name='rows_layer_aed_')
    # print("aed done")
    # print("set comp")
    # plot_mae_comparison(csv1_path=xyz_df, csv2_path=aed_df, labels=("a) Rows XYZ only (Rand)", "b) Rows AED only (Rand)"), split='test', csvFlag=False, show=False)
    # plot_mae_comparison(csv1_path=xyz_df, csv2_path=aed_df, labels=("a) Rows (Cross) XYZ only (Rand)", "b) Rows (Cross) AED only (Rand)"), split='cross', csvFlag=False, show=False)

    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    # # # chosen_features=features
    # spiral_xyz_df = run_comparisons(cross_dataset, dataset, features, target,random_states=states, save_name='spiral_layer_xyz_')
    # print("xyz done")
    # print("set comp")
    # features = ['distance_to_tower', 'azimuth', 'elevation_angle']
    # # chosen_features=features
    # spiral_aed_df = run_comparisons(cross_dataset, dataset, features, target, random_states=states, save_name='spiral_layer_aed_')
    # print("aed done")
    # print("set comp")
    # plot_mae_comparison(csv1_path=spiral_xyz_df, csv2_path=spiral_aed_df, labels=("a) Spiral XYZ only (Rand)", "b) Spiral AED only (Rand)"), split='test', csvFlag=False, show=True)
    # plot_mae_comparison(csv1_path=spiral_xyz_df, csv2_path=spiral_aed_df, labels=("a) Spiral (Cross) XYZ only (Rand)", "b) Spiral (Cross) AED only (Rand)"), split='cross', csvFlag=False, show=True)
    
    # TIMESERIES
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'timestamp_numeric']
    r_ann = 'ROWS-ANN'
    r_annts = 'ROWS-ANN-TS' 
    dataset = cross_dataset
    # res_r_ann = mlp_point_by_point(dataset=dataset, method=r_ann)
    # res_r_annts = mlp_point_by_point(dataset=dataset, features=features, method=r_annts)
    # plot_point_by_point_comparison(csv1_path=res_r_ann, csv2_path=res_r_annts, labels=("a) Rows", "b) Rows as Timeseries"), csvFlag=False)
    dataset['drone_timestamp'] = pd.to_datetime(dataset['drone_timestamp'])
    dataset['timestamp_numeric'] = dataset['drone_timestamp'].astype(np.int64) // 10**9  # seconds since epoch
    xyz_ts_df = run_comparisons(dataset, dataset, features, target, random_states=states, save_name='spiral_layer_xyz_ts_timesplit', split_type="temp")
    print("set comp")
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL']
    timesplit_xyz_ts_df = run_comparisons(dataset, dataset, features, target, random_states=states, save_name='spiral_layer_xyz_timesplit', split_type="temp")
    print("set comp")
    plot_mae_comparison(csv1_path=timesplit_xyz_ts_df, csv2_path=xyz_ts_df, labels=("a) Spiral XYZ", "b) Spiral XYZ Timeseries"), split='test', csvFlag=False)
    





    tuner = RandomSearch(
        build_mlp_tuner,
        objective='val_loss',  # Optimize for lowest validation loss
        max_trials=10,  # Try 10 different hyperparameter sets
        executions_per_trial=2,  # Average over 2 runs per set
        directory='mlp_tuning',  # Folder to store results
        project_name='signal_prediction'
    )

    # Load datasets
    rows = "CONF/cleaned_rows.csv"  # First dataset (e.g., straight-line path)
    spiral = "CONF/cleaned_spiral.csv"  # Second dataset (spiral path)


    dataset = pd.read_csv(spiral)  # Load first dataset
    cross_dataset = pd.read_csv(rows)  # Load second dataset (unseen locations)

    dataset = add_features(dataset)
    cross_dataset = add_features(cross_dataset)
    
    # Select features (MUST match training setup)
    # features = ['distance_to_tower', 'azimuth', 'elevation_angle', 
    #             'gps.lat', 'gps.lon', 'altitudeAMSL']

    # Split first dataset into train and test sets (ensuring test data is from same dataset)
    train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)

    # Features and target variable
    # features = ['gps.lat', 'gps.lon', 'altitudeAMSL']
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', # add in tower feats
                'gps.lat', 'gps.lon', 'altitudeAMSL']
    target = 'rssi'

    # Train data (from first dataset)
    X_train = train_set[features].values
    y_train = train_set[target].values

    # Test data (from first dataset)
    X_test = test_set[features].values
    y_test = test_set[target].values
    tuning = False
    if tuning:
        tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64)
        best_hps = tuner.get_best_hyperparameters(num_trials=5)[0]

        print(f"""
        Best Hyperparameters:
        - Number of Layers: {best_hps.get('num_layers')}
        - Neurons per Layer: {best_hps.get('num_neurons')}
        - Dropout Rate: {best_hps.get('dropout_rate')}
        - Learning Rate: {best_hps.get('learning_rate')}
        """)
        # Best Hyperparameters for MSE LOSS:
        # - Number of Layers: 2
        # - Neurons per Layer: 256
        # - Dropout Rate: 0.5
        # - Learning Rate: 0.01


    
    #  Set seeds for reproducibility
    import tensorflow as tf
    import numpy as np
    import random
    import os
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Completely unseen test data (from second dataset)
    cross_X_test = cross_dataset[features].values
    cross_Y_test = cross_dataset[target].values

    runs =[1,2,3,4,5,6,7,8,9,10]
    same_avg_mae = []
    same_avg_rsme = []
    cross_avg_mae = []
    cross_avg_rsme = []
    # for run in runs:

    # Train the model on dataset1
    mlp_model = run_mlp(input_dim=len(features), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=1000)

    # Evaluate on test set from dataset1
    test_mae, test_rmse = evaluate_mlp(model=mlp_model, X_test=X_test, y_test=y_test)
    print(f"Test Set (Same Path): MAE {test_mae}, RMSE {test_rmse}")
    # same_avg_mae.append(test_mae)
    # same_avg_rsme.append(test_rmse)

    # Evaluate on completely unseen test set (dataset2, different path)
    cross_mae, cross_rmse = evaluate_mlp(model=mlp_model, X_test=cross_X_test, y_test=cross_Y_test)
    print(f"Cross-Test Set (Different Path): MAE {cross_mae}, RMSE {cross_rmse}")
    # cross_avg_mae.append(test_mae)
    # cross_avg_rsme.append(test_rmse)

    # print(f"Same layer {same_avg_mae, same_avg_rsme}, Cross layer {cross_avg_mae, cross_avg_mae}")

        # # new_method(input_df=pd.read_csv(file_name3), model_name="perim")

    # # plot_cross_all_props()

    # show_res = False
    # # --- Run the Function ---
    # # saved_results = pd.read_csv("comparison_random_results.csv")
    # # plot_results(saved_results, "Single Layer Performance")
    #  = "CONF/cleaned_rows.csv"
    # rows = "CONF/cleaned_spiral.csv"
    # ts = [0.1, 0.2, 0.3, 0.4, 0.5]
    # cross_ress = []
    # for t in ts:
    #     res = test_cross_pretrained(file_name=rows, test_size=t)
    #     res.to_csv(f"compiled_cross_ds_unseen({t}).csv")



  

    # # results_df = compare_all_methods(file_name=file_name)
    # # run on baseline 
    # # dataset = pd.read_csv(file_name)
    # # plot_results(results_df, "Single Layer Performance")
    
    # # print("DONE!!!!!!!!!")
    
    # # print("KRIG")
    # # krig_df = do_Krig(df=dataset)
    # # print("ENS")
    # # ens_df = do_Ensemble(df=dataset)
    # # print("TRA")

    # # run on transformer method
    dataset = pd.read_csv(spiral)
    dataset2 = pd.read_csv(rows)
    # # cross_df = pd.concat([dataset, dataset2], axis=0)
    # test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for test_size in test_sizes:
    #     train_df, test_df = train_test_split(dataset2, test_size=test_size, random_state=42)
    #     cross_df = pd.concat([dataset, test_df], axis=0)
    # new_method(input_df=dataset)

    new_method()

    comparison_rsrp = []
    error_data_block = []
    trans_perfs = []
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    helpers_percentage = 1.0
    runs = 5
    for run in range(runs):    
        for test_size in test_sizes:
            print(f"Run {run} - Test size {test_size}")
            if len(rand_states) < 2:
                raise ValueError("rand_states must contain at least two values (current + next states).")

            current_index = rand_states[0]  # Current position in the list
            random_state = rand_states[current_index]  # Get the random state
            # print(f"Random states index: {current_index} with value: {random_state}")

            # Move to the next random state (wrap around if at the end)
            next_index = (current_index + 1) % (len(rand_states) - 1)
            rand_states[0] = next_index  # Update index in config


            # # uncomment for cross layer
            train_df, test_df = train_test_split(dataset2, test_size=test_size, random_state=random_state)
            cross_df = pd.concat([dataset, test_df], axis=0)

            # test_df = train_df
            # train_df = cross_df

            # # uncomment for 1 layer
            # train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_state)
            # trans_df = configure_for_transformer(train_df=train_df, test_df=test_df)


            # print(len(dataset), len(trans_df))
            # help
            # train_helpers, test_helpers, train_pure_spatial, test_pure_spatial = create_train_test_splits_helpers(dataset, test_size=test_size)
            # print(f"Current test size{test_size}, doing pure spatial with tst-tr : {len(train_pure_spatial)}-{len(test_df)}")

            # IDW
            rand_mae, _, block_mae, _ = do_IDW(test_size, train_df, test_df, train_df, test_df)
            comparison_rsrp.append(["IDW", test_size, rand_mae])
            error_data_block.append(["IDW", test_size, block_mae])

            # Kriging
            rand_mae, _, block_mae, _ = do_Krig(test_size, train_df, test_df, train_df, test_df)
            comparison_rsrp.append(["Kriging", test_size, rand_mae])
            error_data_block.append(["Kriging", test_size, block_mae])

            # Ensemble Models
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(test_size, train_df, test_df, train_df, test_df)
            comparison_rsrp.append(["Ensemble", test_size, ens_rand_mae])

            # mlp
            
            mlp_model = train_mlp(train_df) #random split
            unet_rand_mae, _ = evaluate_unet(mlp_model, test_df)  # Evaluate on random test set

            

            # comparison_rsrp.append(["U-Net", test_size, unet_rand_mae])



            res = new_method(input_df=train_df)
            res_mae_rsrp = mean_absolute_error(res["true_rsrp"], res["predicted_rsrp"])
            res_mae_rsrq = mean_absolute_error(res["true_rsrq"], res["predicted_rsrq"])
            res_mae_sinr = mean_absolute_error(res["true_sinr"], res["predicted_sinr"])

            comparison_rsrp.append([f"Transformer", test_size, res_mae_rsrp])
            

            trans_perfs.append([test_size, res_mae_rsrp, res_mae_rsrq, res_mae_sinr])

        
        print(f"MAE RSRP - {res_mae_rsrp}")

        # # with help
        # res = new_method(input_df=train_helpers)
        # print(f"Current test size{test_size}, doing spatial + helpers with tst-tr : {len(train_helpers)}-{len(test_df)}")
        # res_mae_rsrp = mean_absolute_error(res["true_rsrp"], res["predicted_rsrp"])
        # error_data_random.append([f"Transformer(H-{helpers_percentage})", test_size, res_mae_rsrp])
        # print(f"Help - {res_mae_rsrp}")
    
    comparison_rsrp = pd.DataFrame(comparison_rsrp, columns=["Method", "Test Size", "MAE"])
    comparison_rsrp.to_csv('latest-cross-comparison-rsrp.csv', index=False)
    trans_perfs = pd.DataFrame(trans_perfs, columns=["Test Size", "RSRP-MAE", "RSRQ-MAE", "SINR-MAE"])
    trans_perfs.to_csv('latest-cross-trans-sigs.csv', index=False)