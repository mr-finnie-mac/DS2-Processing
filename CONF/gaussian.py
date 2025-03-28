import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import config

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# for presenting splatts after flattening, revesres list to covariece matrix
def preprocess_gaussian_data(data):
    """
    Ensures the covariance matrices are in a valid 2x2 format for visualization.
    Args:
        data (pd.DataFrame): DataFrame with a 'covariance' column.
    Returns:
        Processed DataFrame with reshaped covariance.
    """
    processed_data = data.copy()

    def reshape_cov(cov_list):
        try:
            # Take the first 4 elements and reshape into a 2x2 matrix
            cov_matrix = np.array(cov_list[:4]).reshape(2, 2)
            return cov_matrix
        except:
            return np.array([[1, 0], [0, 1]])  # Default identity matrix if error

    processed_data["covariance"] = processed_data["covariance"].apply(reshape_cov)
    return processed_data

def plot_gaussian_splats(data, tower_location):
    """
    Plots the Gaussian splat footprints using covariance information.

    Args:
        data (pd.DataFrame): DataFrame containing 'gps.lon', 'gps.lat', and 'covariance'.
        tower_location (tuple): (lat, lon) of the cellular tower.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Scatter the Gaussian centers
    sc = ax.scatter(data["gps.lon"], data["gps.lat"], c=data["rssi"], cmap="coolwarm", edgecolors="black", alpha=0.8)
    plt.colorbar(sc, label="RSSI Intensity")

    # Iterate through each point and plot Gaussian splat
    for i, row in data.iterrows():
        lon, lat = row["gps.lon"], row["gps.lat"]
        
        # Print first covariance row for debugging
        cov_raw = np.array(row["covariance"])
        print(f"Index {i} Covariance: {cov_raw}")

        try:
            if len(cov_raw) == 10:  # 10D covariance case
                # Assume first 6 elements are the diagonal and off-diagonal elements in a 3x3 matrix
                cov_matrix = np.array([
                    [cov_raw[0], cov_raw[1], cov_raw[2]],
                    [cov_raw[1], cov_raw[3], cov_raw[4]],
                    [cov_raw[2], cov_raw[4], cov_raw[5]]
                ])

            else:
                print(f"Unexpected covariance shape at index {i}: {cov_raw}")
                continue  # Skip this point if covariance shape is wrong

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Ensure eigenvalues are non-negative (fix sqrt issue)
            eigenvalues = np.maximum(eigenvalues, 0)

            # Compute ellipse parameters
            width, height = 2 * np.sqrt(eigenvalues)  # Scale factor for visualization
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # Convert to degrees

            # Create and add the ellipse (Gaussian footprint)
            ellipse = Ellipse(
                xy=(lon, lat), width=width, height=height, angle=angle,
                edgecolor="black", facecolor="none", linestyle="dashed", linewidth=1
            )
            ax.add_patch(ellipse)

        except Exception as e:
            print(f"Error processing covariance at index {i}: {e}")

    # Plot the tower location
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="*", s=300, label="Tower")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Gaussian Splat Footprints")
    ax.legend()
    plt.grid(True, linestyle="dotted")

    # Save as PDF
    plt.savefig("gaussian_splats.pdf", format="pdf", bbox_inches="tight", dpi=300)
    print("Figure saved as gaussian_splats.pdf")

    plt.show()

def adaptively_cluster_points(df, eps=0.1, min_samples=5):
    """
    Use DBSCAN to cluster points adaptively based on spatial density.
    """
    coords = df[['gps.lat', 'gps.lon', 'altitudeAMSL']].values  # Spatial coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_  # Assign cluster labels
    # print(f"CLUSTERS:{df['cluster']}")
    return df

def compute_anisotropic_covariance(data, fixed_dim=10):
    """Computes anisotropic covariance and ensures fixed dimensionality."""
    
    points = data[["gps.lat", "gps.lon", "altitudeAMSL"]].values
    distances = squareform(pdist(points))  # Pairwise distances
    
    # Compute covariance (example: using exponential decay)
    covariances = np.exp(-distances / np.max(distances))  

    # Ensure all covariance matrices are same size (fixed_dim)
    covariance_vectors = []
    for row in covariances:
        if len(row) > fixed_dim:
            covariance_vectors.append(row[:fixed_dim])  # Truncate if too long
        else:
            covariance_vectors.append(np.pad(row, (0, fixed_dim - len(row)), mode='constant'))  # Pad if too short
    
    return np.array(covariance_vectors)




def create_gaussian_representation(data):
    """
    Creates Gaussian representation of input data with a single column storing covariance.
    
    Args:
        data: DataFrame with spatial and signal strength information.
    
    Returns:
        DataFrame with Gaussian features and a single covariance column.
    """

    # Compute the covariance matrices
    covariance_matrices = compute_anisotropic_covariance(data)

    # Convert each covariance matrix into a numpy array
    covariance_matrices = [np.array(cov) for cov in covariance_matrices]

    # Convert to DataFrame with a single column
    gaussians_df = data.copy()
    gaussians_df["covariance"] = covariance_matrices  # Store full covariance matrix per row

    print(f"Final Gaussian Representation Columns: {gaussians_df.columns}")
    print("Gaussian Representation Sample:")
    print(gaussians_df.head())

    return gaussians_df






# def create_gaussian_representation(df):
#     """
#     Generate Gaussian splats based on clustered lat, lon, altitude, and signal strength.

#     Args:
#         df (pd.DataFrame): Input DataFrame with lat/lon/localXYZ and signal strength.

#     Returns:
#         List[Tuple]: List of (mean, covariance, avg_signal) tuples representing Gaussians.
#     """
#     # Ensure input is a DataFrame
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError(f"Expected input to be a DataFrame, got {type(df)}")

#     # Required columns
#     required_columns = {"gps.lat", "gps.lon", "altitudeAMSL", "rssi"}
#     missing_columns = required_columns - set(df.columns)
#     if missing_columns:
#         raise KeyError(f"Missing columns in DataFrame: {missing_columns}")

#     print("All required columns are present.")

#     # Step 1: Cluster the points
#     df = adaptively_cluster_points(df)

#     # Step 2: Compute Gaussian features per cluster
#     gaussian_features = []
#     for cluster_id in df['cluster'].unique():
#         cluster_data = df[df['cluster'] == cluster_id]
        
#         if len(cluster_data) > 1:
#             mean = cluster_data[['gps.lat', 'gps.lon', 'altitudeAMSL']].mean().values
#             covariance = np.cov(cluster_data[['gps.lat', 'gps.lon', 'altitudeAMSL']].values.T)
#         else:
#             mean = cluster_data[['gps.lat', 'gps.lon', 'altitudeAMSL']].values[0]
#             covariance = np.eye(3) * 1e-6  # Small default covariance for single points

#         avg_signal = cluster_data["rssi"].mean()
#         gaussian_features.append((mean, covariance, avg_signal))

#     # Convert to DataFrame
#     # Add back original columns (taking the first row of each cluster)
#     original_columns = df.columns  # Save original column names
#     gaussian_df = pd.DataFrame(gaussian_features, columns=["mean", "covariance", "rssi_mean"])

#     # Merge with original DataFrame to retain required columns
#     gaussian_df = gaussian_df.join(df.groupby("cluster").first().reset_index(), rsuffix="_original")

#     # Ensure the correct columns are present
#     gaussian_df = gaussian_df[["mean", "covariance", "rssi_mean"] + list(original_columns)]

#     return gaussian_df  # Return a proper DataFrame


