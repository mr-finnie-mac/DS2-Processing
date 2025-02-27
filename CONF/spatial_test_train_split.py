import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# # Load dataset (modify path as needed)
# df = pd.read_csv("CONF/cleaned_spiral.csv")

# # Extract relevant columns (position + signal strength)
# position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
# signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]

# # Ensure no NaNs before splitting
# df = df.dropna(subset=position_cols + signal_cols)

# # Convert to NumPy array for spatial operations
# positions = df[position_cols].values

# Random Point Subsampling
def random_point_subsampling(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df

# train_random, test_random = random_point_subsampling(df, test_size=0.2)


# Spatial Block Splitting
def spatial_block_split(df, n_clusters=10, test_fraction=0.2, positions = 0):
    """Splits data into train-test sets based on spatial clusters"""
    
    # Apply KMeans clustering to group points into regions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(positions)

    # Randomly select clusters for testing
    unique_clusters = df["cluster"].unique()
    test_clusters = np.random.choice(unique_clusters, size=int(len(unique_clusters) * test_fraction), replace=False)

    # Split data based on clusters
    train_df = df[~df["cluster"].isin(test_clusters)]
    test_df = df[df["cluster"].isin(test_clusters)]

    return train_df.drop(columns=["cluster"]), test_df.drop(columns=["cluster"])

# train_block, test_block = spatial_block_split(df, n_clusters=10, test_fraction=0.2)



def visualize_splits(train_random, test_random, train_block, test_block):
    plt.figure(figsize=(12, 5))

    # Count unique points based on lat and lon
    random_train_unique = train_random.drop_duplicates(subset=["gps.lat", "gps.lon"]).shape[0]
    random_test_unique = test_random.drop_duplicates(subset=["gps.lat", "gps.lon"]).shape[0]
    block_train_unique = train_block.drop_duplicates(subset=["gps.lat", "gps.lon"]).shape[0]
    block_test_unique = test_block.drop_duplicates(subset=["gps.lat", "gps.lon"]).shape[0]

    # Random Split
    plt.subplot(1, 2, 1)
    plt.scatter(train_random["gps.lat"], train_random["gps.lon"], color="blue", label=f"Train ({random_train_unique})", alpha=1, s=15)
    plt.scatter(test_random["gps.lat"], test_random["gps.lon"], color="green", label=f"Test ({random_test_unique})", alpha=1, s=15)
    plt.title(f"Random Split: Train={random_train_unique}, Test={random_test_unique}")
    plt.xlabel("RTK Latitude")
    plt.ylabel("RTK Longitude")
    plt.legend()

    # Block Split
    plt.subplot(1, 2, 2)
    plt.scatter(train_block["gps.lat"], train_block["gps.lon"], color="blue", label=f"Train ({block_train_unique})", alpha=1,  s=15)
    plt.scatter(test_block["gps.lat"], test_block["gps.lon"], color="green", label=f"Test ({block_test_unique})", alpha=1, s=15)
    plt.title(f"Block Split: Train={block_train_unique}, Test={block_test_unique}")
    plt.xlabel("RTK Latitude")
    plt.ylabel("RTK Longitude")
    plt.legend()

    plt.tight_layout()
    plt.show()



def perform_splits(filename = "placeholder.csv", this_test_size=0.2, this_n_clusters = 9, this_test_fraction = 0.2):
    # Load dataset (modify path as needed)
    this_df = pd.read_csv(filename)

    # Extract relevant columns (position + signal strength)
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]

    # Ensure no NaNs before splitting
    this_df = this_df.dropna(subset=position_cols + signal_cols)

    # Convert to NumPy array for spatial operations
    this_positions = this_df[position_cols].values

    train_random, test_random = random_point_subsampling(df=this_df, test_size=this_test_size)
    train_block, test_block = spatial_block_split(df=this_df, n_clusters=this_n_clusters, test_fraction=this_test_fraction, positions=this_positions)

    # Summary of splits
    print(f"Random Split -> Train: {len(train_random)}, Test: {len(test_random)}")
    print(f"Block Split  -> Train: {len(train_block)}, Test: {len(test_block)}")

    # visualize_splits(train_random, test_random, train_block, test_block) # if want to see the distibution of test and train points


    return train_random, test_random, train_block, test_block


# train_random, test_random, train_block, test_block = perform_splits(filename="CONF/cleaned_spiral.csv",  this_test_size=0.2, this_n_clusters = 10, this_test_fraction = 0.2)