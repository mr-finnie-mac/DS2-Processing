import numpy as np
from scipy.spatial import cKDTree
from spatial_test_train_split import perform_splits
import config
import pickle
import numpy as np
import pandas as pd
from config import *

def idw_interpolation(train_df, test_df, position_cols, signal_col, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation using all known points.

    Args:
        train_df (pd.DataFrame): DataFrame with known points.
        test_df (pd.DataFrame): DataFrame with unknown points (to predict).
        position_cols (list): List of column names for spatial coordinates (e.g., ['x', 'y', 'z']).
        signal_col (str): Name of the signal strength column to interpolate.
        power (float): Inverse distance power parameter (higher = more local influence).

    Returns:
        np.ndarray: Interpolated signal strength values for test_df.
    """
    print("idw-position_cols:", position_cols)
    # Extract coordinates and signal values
    train_positions = train_df[position_cols].values
    test_positions = test_df[position_cols].values
    train_signals = train_df[signal_col].values

    interpolated_values = []

    for test_point in test_positions:
        # Compute distances from the test point to all train points
        dists = np.linalg.norm(train_positions - test_point, axis=1) + 1e-9  # avoid division by zero

        # Compute inverse distance weights
        weights = 1 / (dists ** power)
        weights /= weights.sum()  # normalize weights

        # Compute weighted sum of signal values
        interpolated_value = np.sum(weights * train_signals)
        interpolated_values.append(interpolated_value)

    return np.array(interpolated_values)



def perform_IDW(target = "rsrp", this_filename = "placeholder.py", test_size = 0.2, test_fraction = 0.2, train_random=0, test_random=0, train_block=0, test_block=0, position_cols=[]):
    # train_random, test_random, train_block, test_block = perform_splits(filename=this_filename,  this_test_size=test_size, this_n_clusters = 10, this_test_fraction = test_fraction)

    # RANDOM Example usage with RSRP signal
    # position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = target # Can change to 'rssi', 'sinr', etc.

    random_test_df = test_random.copy()  # Keep original test data for comparison
    random_test_df["idw_predicted_"+target] = idw_interpolation(train_random, test_random, position_cols, signal_col)
    # print(random_test_df[["rsrp", "idw_predicted_rsrp"]])  # Compare actual vs. predicted


    # BLOCK Example usage with RSRP signal
    # position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = target  # Can change to 'rssi', 'sinr', etc.

    block_test_df = test_block.copy()  # Keep original test data for comparison
    block_test_df["idw_predicted_"+target] = idw_interpolation(train_block, test_block, position_cols, signal_col)

    # print(block_test_df[["rsrp", "idw_predicted_rsrp"]])  # Compare actual vs. predicted
    # Save the known training points (lat, lon, rsrp)
    idw_data = random_test_df[["gps.lat", "gps.lon", target]].copy()

    with open("idw_data.pkl", "wb") as f:
        pickle.dump(idw_data, f)
    return random_test_df, block_test_df
