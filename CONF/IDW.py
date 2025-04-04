import numpy as np
from scipy.spatial import cKDTree
from spatial_test_train_split import perform_splits
import config
import pickle
def idw_interpolation(train_df, test_df, position_cols, signal_col, power=2, k=5):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.

    Args:
        train_df: DataFrame with known points.
        test_df: DataFrame with unknown points (to predict).
        position_cols: List of column names for spatial coordinates.
        signal_col: Name of the signal strength column to interpolate.
        power: Inverse distance power parameter (higher = more local influence).
        k: Number of nearest neighbors to use.

    Returns:
        Interpolated values for test_df.
    """
    # Convert coordinates and signal to NumPy arrays
    train_positions = train_df[position_cols].values
    test_positions = test_df[position_cols].values
    train_values = train_df[signal_col].values

    # Build a KDTree for fast nearest-neighbor search
    tree = cKDTree(train_positions)

    # Find k nearest neighbors for each test point
    distances, indices = tree.query(test_positions, k=k)

    # Compute weights (inverse distance)
    weights = 1 / (distances ** power + 1e-9)  # Add small value to prevent division by zero
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights

    # Compute weighted average of known values
    interpolated_values = np.sum(weights * train_values[indices], axis=1)

    return interpolated_values


def perform_IDW(target = "rsrp", this_filename = "placeholder.py", test_size = 0.2, test_fraction = 0.2, train_random=0, test_random=0, train_block=0, test_block=0):
    # train_random, test_random, train_block, test_block = perform_splits(filename=this_filename,  this_test_size=test_size, this_n_clusters = 10, this_test_fraction = test_fraction)

    # RANDOM Example usage with RSRP signal
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = target # Can change to 'rssi', 'sinr', etc.

    random_test_df = test_random.copy()  # Keep original test data for comparison
    random_test_df["idw_predicted_"+target] = idw_interpolation(train_random, test_random, position_cols, signal_col)
    # print(random_test_df[["rsrp", "idw_predicted_rsrp"]])  # Compare actual vs. predicted


    # BLOCK Example usage with RSRP signal
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = target  # Can change to 'rssi', 'sinr', etc.

    block_test_df = test_block.copy()  # Keep original test data for comparison
    block_test_df["idw_predicted_"+target] = idw_interpolation(train_block, test_block, position_cols, signal_col)

    # print(block_test_df[["rsrp", "idw_predicted_rsrp"]])  # Compare actual vs. predicted
    # Save the known training points (lat, lon, rsrp)
    idw_data = random_test_df[["gps.lat", "gps.lon", target]].copy()

    with open("idw_data.pkl", "wb") as f:
        pickle.dump(idw_data, f)
    return random_test_df, block_test_df
