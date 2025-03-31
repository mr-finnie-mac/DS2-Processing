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

def new_method(file_name="placeholder", input_df = []):
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

    torch.save(model.state_dict(), "transformer_model.pth")

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


    

if __name__ == '__main__':
    show_res = False
    # --- Run the Function ---
    # saved_results = pd.read_csv("comparison_random_results.csv")
    # plot_results(saved_results, "Single Layer Performance")
    file_name = "CONF/cleaned_rows.csv"

    # results_df = compare_all_methods(file_name=file_name)
    # run on baseline 
    # dataset = pd.read_csv(file_name)
    # plot_results(results_df, "Single Layer Performance")
    
    # print("DONE!!!!!!!!!")
    
    # print("KRIG")
    # krig_df = do_Krig(df=dataset)
    # print("ENS")
    # ens_df = do_Ensemble(df=dataset)
    # print("TRA")

    # run on transformer method
    dataset = pd.read_csv(file_name)
    
    comparison_rsrp = []
    error_data_block = []
    trans_perfs = []
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    helpers_percentage = 1.0
    runs = 1
    for run in range(runs):    
        print(f"Run {run}")
        for test_size in test_sizes:
            print(f"Test size {test_size}")
            if len(rand_states) < 2:
                raise ValueError("rand_states must contain at least two values (current + next states).")

            current_index = rand_states[0]  # Current position in the list
            random_state = rand_states[current_index]  # Get the random state
            # print(f"Random states index: {current_index} with value: {random_state}")

            # Move to the next random state (wrap around if at the end)
            next_index = (current_index + 1) % (len(rand_states) - 1)
            rand_states[0] = next_index  # Update index in config

            train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_state)
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

            # UNET
            unet_model = train_unet(train_df, epochs = 100) #random spli
            unet_rand_mae, _ = evaluate_unet(unet_model, test_df)  # Evaluate on random test set

            comparison_rsrp.append(["U-Net", test_size, unet_rand_mae])



            res = new_method(input_df=train_df)
            res_mae_rsrp = mean_absolute_error(res["true_rsrp"], res["predicted_rsrp"])
            res_mae_rsrq = mean_absolute_error(res["true_rsrq"], res["predicted_rsrq"])
            res_mae_sinr = mean_absolute_error(res["true_sinr"], res["predicted_sinr"])

            comparison_rsrp.append([f"Transformer", test_size, res_mae_rsrp])
            

            trans_perfs.append([test_size, res_mae_rsrp, res_mae_rsrq, res_mae_sinr])
        # print(f"MAE RSRP - {res_mae_rsrp}")

        # # with help
        # res = new_method(input_df=train_helpers)
        # print(f"Current test size{test_size}, doing spatial + helpers with tst-tr : {len(train_helpers)}-{len(test_df)}")
        # res_mae_rsrp = mean_absolute_error(res["true_rsrp"], res["predicted_rsrp"])
        # error_data_random.append([f"Transformer(H-{helpers_percentage})", test_size, res_mae_rsrp])
        # print(f"Help - {res_mae_rsrp}")

    
    comparison_rsrp = pd.DataFrame(comparison_rsrp, columns=["Method", "Test Size", "MAE"])
    comparison_rsrp.to_csv('comparison-rsrp.csv', index=False)
    trans_perfs = pd.DataFrame(trans_perfs, columns=["Test Size", "RSRP-MAE", "RSRQ-MAE", "SINR-MAE"])
    trans_perfs.to_csv('trans-sigs.csv', index=False)

# # Example function for loading models
# def load_pytorch_model(model_class, model_path):
#     model = model_class(input_dim=10)  # Make sure input_dim is the same as during training
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set to eval mode
#     return model

# def load_keras_model(model_path):
#     return load_model(model_path)

# # Example for data preprocessing (you can adjust based on your needs)
# def preprocess_new_data(df, feature_columns, scaler=None):
#     features = df[feature_columns]
    
#     if scaler:
#         features = scaler.transform(features)
#     else:
#         scaler = StandardScaler()
#         features = scaler.fit_transform(features)
    
#     return torch.tensor(features, dtype=torch.float32), scaler

# # Model evaluation function
# def evaluate_model(model, X_new, y_true, model_type='pytorch'):
#     if model_type == 'pytorch':
#         with torch.no_grad():
#             y_pred = model(X_new).numpy()
#     elif model_type == 'keras':
#         y_pred = model.predict(X_new)
    
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
    
#     return mse, mae



# # Load all models (adjust paths accordingly)
# models = {
#     'unet': load_keras_model('unet_model.h5'),  # If using Keras for U-Net
#     'transformer': load_pytorch_model(SignalQualityTransformer, 'transformer_model.pth'),
#     # Add other models as needed (IDW, RIGGING, Ensemble)
# }

# # Load new dataset layer (e.g., 20m altitude dataset)
# new_data = pd.read_csv('new_dataset_20m.csv')

# # Feature engineering
# # Tower distance
# distances = compute_distance_to_tower(new_data, tower_location=tower_position)/1000
# # train['distance_to_tower'] = distances
# new_data.loc[:, 'distance_to_tower'] = distances
# # Tower direction
# _, elevation_angle = compute_tower_direction_local(new_data)
# azimuths,_ = compute_tower_direction(new_data, tower_location=tower_position)
# # train['azimuth'] = azimuths # add az to input
# new_data.loc[:, 'azimuth'] = azimuths
# # train['elevation_angle'] = elevation_angle # add e angle to input
# new_data.loc[:, 'elevation_angle'] = elevation_angle
# print(distances, azimuths, elevation_angle)

# # Assuming you have a list of feature columns and target columns
# feature_columns = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'pressure', 'temperature', 'humidity', 'gas_resistance']  # Adjust as per your data
# target_columns = ['rsrp', 'rsrq', 'rssi', 'sinr']

# # Separate features and targets from the new dataset
# X_new = new_data[feature_columns]
# y_true = new_data[target_columns]

# # Preprocess data for both PyTorch and Keras models
# X_new_tensor, scaler = preprocess_new_data(X_new, feature_columns)

# # Evaluate all models
# for model_name, model in models.items():
#     if model_name == 'transformer':
#         mse, mae = evaluate_model(model, X_new_tensor, y_true.values, model_type='pytorch')
#     else:
#         mse, mae = evaluate_model(model, X_new, y_true.values, model_type='keras')
    
#     print(f"{model_name} Model - MSE: {mse}, MAE: {mae}")


    
    # error_data_block.append(["IDW", test_size, block_mae])
    # create_train_test_splits_helpers(dataset)
    # new_method("CONF/cleaned_rows.csv")
    # new_method("CONF/cleaned_spiral2.csv")