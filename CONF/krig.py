from pykrige.ok import OrdinaryKriging
from spatial_test_train_split import perform_splits
import numpy as np
import config
import pickle

def kriging_interpolation(train_df, test_df, position_cols, signal_col):
    """
    Perform Ordinary Kriging interpolation.

    Args:
        train_df: DataFrame with known points.
        test_df: DataFrame with unknown points.
        position_cols: List of column names for spatial coordinates.
        signal_col: Name of the signal strength column to interpolate.

    Returns:
        Interpolated values for test_df.
    """
    # Extract position and signal data
    train_positions = train_df[position_cols].values.T  # PyKrige needs transposed coordinates
    test_positions = test_df[position_cols].values.T
    train_values = train_df[signal_col].values

    if len(train_df) < 10:
        print("Warning: Not enough training data for Kriging!")
        return np.full(len(test_df), np.nan)  # Return NaNs instead of halting crash


    # Ordinary Kriging model with nugget effect for stability
    OK = OrdinaryKriging(
        train_positions[0], train_positions[1], train_values,# X, Y, Signal
        variogram_model="gaussian", # "spherical", "linear", "power", "exponential" <-good for abrupt change, "gaussian" <- good for some sharp changes
        verbose=False,
        enable_plotting=False
    )

    # Perform interpolation
    interpolated_values, _ = OK.execute("points", test_positions[0], test_positions[1])

    return interpolated_values

def perform_Krig(this_filename = "placeholder.py", test_size = 0.2, test_fraction = 0.2, train_random=0, test_random=0, train_block=0, test_block=0):
    # train_random, test_random, train_block, test_block = perform_splits(filename=this_filename,  this_test_size=test_size, this_n_clusters = 10, this_test_fraction = test_fraction)

    # rsrp example
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = "rsrp"  # Can change to 'rssi', 'sinr', etc.

    random_test_df = test_random.copy()  # Keep original test data for comparison
    # Example usage with RSRP signal
    random_test_df["kriging_predicted_rsrp"] = kriging_interpolation(train_random, test_random, position_cols[:2], signal_col)

    # print(test_df[["rsrp", "kriging_predicted_rsrp"]])  # Compare actual vs. predicted

    # rsrp example
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_col = "rsrp"  # Can change to 'rssi', 'sinr', etc.

    block_test_df = test_block.copy()  # Keep original test data for comparison
    # Example usage with RSRP signal
    block_test_df["kriging_predicted_rsrp"] = kriging_interpolation(train_block, test_block, position_cols[:2], signal_col)
    
    krig_data = random_test_df[["gps.lat", "gps.lon", "rsrp"]].copy()

    with open("krig_data.pkl", "wb") as f:
        pickle.dump(krig_data, f)
    # print(random_test_df[["rsrp", "kriging_predicted_rsrp"]])  # Compare actual vs. predicted
    return random_test_df, block_test_df
