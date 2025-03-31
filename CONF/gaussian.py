import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import config
from transformer import compute_distance_to_tower, compute_tower_direction, compute_sinr_weighted_rssi
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches


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

def adaptively_cluster_points(df, eps=0.00001, min_samples=5):
    """
    Use DBSCAN to cluster points adaptively based on spatial density.
    """
    coords = df[['gps.lat', 'gps.lon', 'altitudeAMSL']].values  # Spatial coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_  # Assign cluster labels
    # print(f"CLUSTERS:{df['cluster']}")
    return df

# def create_gaussians_for_clusters(df):
#     """
#     Generates Gaussian representations for each cluster separately.
#     """
#     gaussians = []
#     for cluster_id in df['cluster'].unique():
#         if cluster_id == -1: continue  # Ignore noise points
#         cluster_data = df[df['cluster'] == cluster_id]
#         gaussians.append(create_gaussian_representation(cluster_data))

#     return pd.concat(gaussians, ignore_index=True) if gaussians else pd.DataFrame()

def cluster_and_assign_means(df, eps=0.5, min_samples=5, fixed_dim=10):
    """
    Clusters points using DBSCAN and assigns each cluster its covariance representation.
    """
    df = df.copy()
    coords = df[['localPosition.x', 'localPosition.y', 'localPosition.z']].values  # Spatial coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_  # Assign cluster labels

    return df

def create_gaussians_for_clusters(df, fixed_dim=10):
    """
    Generates Gaussian representations (covariance vectors) for each cluster separately.
    """
    gaussians = []
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue  # Ignore noise points
        cluster_data = df[df['cluster'] == cluster_id]
        cov_vector = compute_anisotropic_covariance(cluster_data, fixed_dim)
        gaussians.append({'cluster': cluster_id, 'cov_vector': cov_vector})
    
    return pd.DataFrame(gaussians)
# def cluster_and_assign_means(df, eps=0.5, min_samples=5):
#     """
#     Clusters points using DBSCAN and assigns each cluster its mean position.
#     """
#     df = df.copy()
#     coords = df[['localPosition.x', 'localPosition.y', 'localPosition.z']].values  # Spatial coordinates
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
#     df['cluster'] = clustering.labels_  # Assign cluster labels

#     # Compute mean position per cluster
#     cluster_means = df.groupby('cluster')[['gps.lat', 'gps.lon', 'altitudeAMSL']].mean()

#     # Assign mean position back to each point in the cluster
#     df = df.merge(cluster_means, on='cluster', suffixes=('', '_mean'))
#     df[['gps.lat', 'gps.lon', 'altitudeAMSL']] = df[['gps.lat_mean', 'gps.lon_mean', 'altitudeAMSL_mean']]
#     df.drop(columns=['gps.lat_mean', 'gps.lon_mean', 'altitudeAMSL_mean'], inplace=True)

#     return df


# def create_gaussians_for_clusters(df):
#     """
#     Generates Gaussian representations for each cluster separately.
#     """
#     gaussians = []
#     for cluster_id in df['cluster'].unique():
#         if cluster_id == -1: continue  # Ignore noise points
#         cluster_data = df[df['cluster'] == cluster_id]
#         gaussians.append(create_gaussian_representation(cluster_data))

#     return pd.concat(gaussians, ignore_index=True) if gaussians else pd.DataFrame()



import numpy as np

def compute_anisotropic_covariance(data, fixed_dim=10):
    """Computes mean position and anisotropic covariance, then ensures fixed dimensionality."""
    
    # Extract relevant columns
    pts = data[["gps.lat", "gps.lon", "altitudeAMSL"]].values
    
    # Compute mean position
    mean_pos = np.mean(pts, axis=0)
    
    # Compute 3x3 covariance matrix
    cov_matrix = np.cov(pts, rowvar=False)
    
    # Flatten the covariance matrix to a 1D array
    flat_cov = cov_matrix.flatten()
    
    # Combine mean position and flattened covariance into a single vector
    combined_vector = np.concatenate((mean_pos, flat_cov))
    
    # Ensure the vector is of fixed_dim size
    if len(combined_vector) < fixed_dim:
        padded_vector = np.pad(combined_vector, (0, fixed_dim - len(combined_vector)), mode='constant')
    else:
        padded_vector = combined_vector[:fixed_dim]  # Truncate if necessary
    
    print(f"Mean Position: {mean_pos}")
    print(f"cvariance Matrix:\n{cov_matrix}")
    print(f"final vector fixed {fixed_dim}): {padded_vector}")
    
    return padded_vector




def plot_clusters_with_gaussians(df):
    """
    Plots the mean position and covariance as Gaussian ellipses for each cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue  # Ignore noise points
        cluster_data = df[df['cluster'] == cluster_id]
        mean_pos = np.mean(cluster_data[['gps.lat', 'gps.lon']], axis=0)
        cov_matrix = np.cov(cluster_data[['gps.lat', 'gps.lon']].values, rowvar=False)
        
        # Eigen decomposition to get ellipse axes
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)  # Scale for visualization
        
        # Plot mean position
        ax.scatter(*mean_pos, label=f'Cluster {cluster_id}', s=50)
        
        # Plot covariance as an ellipse
        ellipse = patches.Ellipse(mean_pos, width, height, angle, edgecolor='r', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.legend()
    plt.title("Cluster Mean Positions and Covariance Ellipses")
    plt.show()


def create_gaussian_representation(data):
    """
    Creates Gaussian representation 
    
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



def new_generate_gaussian_features(data, tower_location=(52.60818, 1.542818, 15.2)):
    """
    Extracts Gaussian features and includes cluster IDs for transformer training.
    """

    feature_columns = ['gps.lat', 'gps.lon', 'altitudeAMSL', 
                       'localPosition.x', 'localPosition.y', 'localPosition.z', 
                       'rsrp', 'rsrq', 'rssi', 'sinr']
    
    spatial_signal_features = data[feature_columns].values
    dist_to_tower = compute_distance_to_tower(data, tower_location).values.reshape(-1, 1)
    azimuth, elevation = compute_tower_direction(data, tower_location)
    sinr_weighted_rssi = compute_sinr_weighted_rssi(data).values.reshape(-1, 1)

    azimuth = azimuth.values.reshape(-1, 1)
    elevation = elevation.values.reshape(-1, 1)

    if "covariance" in data.columns:
        covariance_features = np.array([cov.flatten() for cov in data["covariance"].values])
    else:
        covariance_features = np.zeros((data.shape[0], 10))  # Placeholder if missing

    scaler = StandardScaler()
    covariance_features = scaler.fit_transform(covariance_features)

    # Include cluster ID in final dataset
    cluster_ids = data['cluster'].values.reshape(-1, 1)

    features = np.hstack([
        spatial_signal_features,
        sinr_weighted_rssi,
        dist_to_tower,
        azimuth, elevation,
        covariance_features,
        cluster_ids  # Adding cluster ID to the input
    ])

    features = scaler.fit_transform(features)

    return features



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


