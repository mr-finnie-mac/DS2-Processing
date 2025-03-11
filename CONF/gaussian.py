import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import config

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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


def adaptively_cluster_points(df, eps=0.001, min_samples=5):
    """
    Use DBSCAN to cluster points adaptively based on spatial density.
    """
    coords = df[['gps.lat', 'gps.lon', 'altitudeAMSL']].values  # Spatial coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_  # Assign cluster labels
    return df

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


