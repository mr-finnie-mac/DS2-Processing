import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from eval_baselines import *
import config

def calculate_distance_from_origin(df):
    """
    Calculate the Euclidean distance from the origin (0, 0).
    Assumes df has 'x' and 'y' coordinates for each point.
    """
    return np.sqrt(df['x']**2 + df['y']**2)

def evaluate_and_plot_errors(df, predicted_column, true_column):
    """
    Evaluate prediction errors and plot them against the distance from origin.
    """
    # Calculate distance from origin
    df['distance'] = calculate_distance_from_origin(df)

    # Calculate absolute error
    df['error'] = np.abs(df[true_column] - df[predicted_column])

    # Group by distance to find mean error for each distance bin
    distance_bins = np.arange(0, df['distance'].max() + 1, 1)  # Adjust bin size as needed
    df['distance_bin'] = np.digitize(df['distance'], distance_bins)

    # Calculate mean error for each distance bin
    error_by_distance = df.groupby('distance_bin')['error'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(error_by_distance['distance_bin'], error_by_distance['error'], marker='o')
    plt.title('Mean Prediction Error vs Distance from (0, 0)')
    plt.xlabel('Distance from (0, 0)')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.show()

# Example usage with IDW predictions
def evaluate_idw(file_name):
    random_test_df, block_test_df = perform_IDW(this_filename=file_name)
    
    # Evaluate and plot for Random Test
    print("Evaluating IDW Random Test")
    evaluate_and_plot_errors(random_test_df, 'idw_predicted_rsrp', 'rsrp')
    
    # Evaluate and plot for Block Test
    print("Evaluating IDW Block Test")
    evaluate_and_plot_errors(block_test_df, 'idw_predicted_rsrp', 'rsrp')

# Example function call
evaluate_idw("CONF/cleaned_spiral.csv")
