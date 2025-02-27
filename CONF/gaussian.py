import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


def compute_anisotropic_covariance(points, k=5):
    """
    Compute anisotropic covariance matrices for each point based on k-nearest neighbors.
    """
    distances = squareform(pdist(points))  # Pairwise distances
    covariances = []
    
    for i in range(len(points)):
        neighbors = np.argsort(distances[i])[1:k+1]  # Get k nearest neighbors
        local_points = points[neighbors]
        cov_matrix = np.cov(local_points, rowvar=False)  # Covariance matrix
        covariances.append(cov_matrix)
    
    return np.array(covariances)


def adaptively_cluster_points(df, eps=0.001, min_samples=5):
    """
    Use DBSCAN to cluster points adaptively based on spatial density.
    """
    coords = df[['gps.lat', 'gps.lon', 'altitudeAMSL']].values  # Spatial coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_  # Assign cluster labels
    return df

def create_gaussian_representation(df):
    """
    Generate Gaussian splats based on lat, lon, altitude, and signal strength.

    Args:
        df (pd.DataFrame): Input DataFrame with lat/lon/localXYZ and signal strength.

    Returns:
        pd.DataFrame: Processed Gaussian representation.
    """
    
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input to be a DataFrame, got {}".format(type(df)))

    # Required columns
    required_columns = {"gps.lat", "gps.lon", "altitudeAMSL", "localPosition.x", "localPosition.y", "localPosition.z",
                        "rsrp", "rsrq", "rssi", "sinr"}

    # missing_columns = required_columns - set(df.columns)
    # if missing_columns:
    #     raise KeyError(f"Missing columns in DataFrame: {missing_columns}")

    print(" All required columns are present.")

    # Compute Gaussian properties
    covariance = compute_anisotropic_covariance(df[['gps.lat', 'gps.lon', 'altitudeAMSL']].values)
    df = df.copy()  # Make a copy of the DataFrame
    df["covariance"] = list(covariance)


    # Return complete DataFrame
    return df.copy()

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


