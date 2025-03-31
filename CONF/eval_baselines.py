from config import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# 52.60818, 1.542818, altitudeAMSL: 15.2
title_s = 18
label_s = 16
line_w = 6

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "axes.labelsize": 14,
#     "axes.titlesize": 16,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16
# })

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_predictions(y_true, y_pred):
    """
    Compute evaluation metrics: MAE, RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

# # Evaluate IDW
# random_test_df, block_test_df = perform_IDW(this_filename="CONF/cleaned_spiral.csv")
# random_idw_metrics = evaluate_predictions(random_test_df["rsrp"], random_test_df["idw_predicted_rsrp"])
# block_idw_metrics = evaluate_predictions(block_test_df["rsrp"], block_test_df["idw_predicted_rsrp"])
# print("Random IDW Performance:", random_test_df)
# print("Block IDW Performance:", block_test_df)

# random_test_df, block_test_df = perform_Krig(this_filename="CONF/cleaned_spiral.csv",)
# # Evaluate Kriging
# random_kriging_metrics = evaluate_predictions(random_test_df["rsrp"], random_test_df["kriging_predicted_rsrp"])
# block_kriging_metrics = evaluate_predictions(block_test_df["rsrp"], block_test_df["kriging_predicted_rsrp"])

# print("Random IDW Performance:", random_idw_metrics)
# print("Block IDW Performance:", block_idw_metrics)
# print("Random Kriging Performance:", random_kriging_metrics)
# print("Block Kriging Performance:", block_kriging_metrics)
# mae = random_idw_metrics["MAE"]
# print(mae)

def do_IDW(size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0, file_name = "placeholder"):
    # Evaluate IDW
    random_test_df, block_test_df = perform_IDW(this_filename=file_name, test_size=size, test_fraction=size, train_random=this_train_random, test_random=this_test_random, train_block=this_train_block, test_block=this_test_block)
    random_idw_metrics = evaluate_predictions(random_test_df["rsrp"], random_test_df["idw_predicted_rsrp"])
    block_idw_metrics = evaluate_predictions(block_test_df["rsrp"], block_test_df["idw_predicted_rsrp"])
    print("Random IDW Performance:", random_test_df)
    print("Block IDW Performance:", block_test_df)
    random_mae = random_idw_metrics["MAE"]
    random_rmse = random_idw_metrics["RMSE"]
    block_mae = block_idw_metrics["MAE"]
    block_rmse = block_idw_metrics["RMSE"]

    return random_mae, random_rmse, block_mae, block_rmse


def do_Krig(size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0, file_name = "placeholder"):
    # Evaluate IDW
    random_test_df, block_test_df = perform_Krig(this_filename=file_name, test_size=size, test_fraction=size, train_random=this_train_random, test_random=this_test_random, train_block=this_train_block, test_block=this_test_block)
    random_kriging_metrics = evaluate_predictions(random_test_df["rsrp"], random_test_df["kriging_predicted_rsrp"])
    block_kriging_metrics = evaluate_predictions(block_test_df["rsrp"], block_test_df["kriging_predicted_rsrp"])
    print("Random IDW Performance:", random_test_df)
    print("Block IDW Performance:", block_test_df)
    random_mae = random_kriging_metrics["MAE"]
    random_rmse = random_kriging_metrics["RMSE"]
    block_mae = block_kriging_metrics["MAE"]
    block_rmse = block_kriging_metrics["RMSE"]

    return random_mae, random_rmse, block_mae, block_rmse

def do_Ensemble(size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0):
    random_ensemble, block_ensemble = perform_ensemble(size, train_random=this_train_random, test_random=this_test_random, train_block=this_train_block, test_block=this_test_block)
    # Print results
    # RF
    # Access Random Forest results from the nested dictionary
    rf_random_mae = random_ensemble["Random Forest"]["MAE"]
    rf_random_rmse = random_ensemble["Random Forest"]["RMSE"]
    rf_block_mae = block_ensemble["Random Forest"]["MAE"]
    rf_block_rmse = block_ensemble["Random Forest"]["RMSE"]

    xg_random_mae = random_ensemble["XGBoost"]["MAE"]
    xg_random_rmse = random_ensemble["XGBoost"]["RMSE"]
    xg_block_mae = block_ensemble["XGBoost"]["MAE"]
    xg_block_rmse = block_ensemble["XGBoost"]["RMSE"]

    lg_random_mae = random_ensemble["LightGBM"]["MAE"]
    lg_random_rmse = random_ensemble["LightGBM"]["RMSE"]
    lg_block_mae = block_ensemble["LightGBM"]["MAE"]
    lg_block_rmse = block_ensemble["LightGBM"]["RMSE"]

    ens_random_mae = random_ensemble["Ensemble"]["MAE"]
    ens_random_rmse = random_ensemble["Ensemble"]["RMSE"]
    ens_block_mae = block_ensemble["Ensemble"]["MAE"]
    ens_block_rmse = block_ensemble["Ensemble"]["RMSE"]

    print("Performance on Random Split:")
    for model, metrics in random_ensemble.items():
        print(f"{model}: {metrics}")

    print("\nPerformance on Spatial Block Split:")
    for model, metrics in block_ensemble.items():
        print(f"{model}: {metrics}")
    
    print(rf_random_mae, rf_random_rmse, rf_block_mae, rf_block_rmse, xg_random_mae, xg_random_rmse, xg_block_mae, xg_block_rmse, lg_random_mae, lg_random_rmse, lg_block_mae, lg_block_rmse, ens_random_mae, ens_random_rmse, ens_block_mae, ens_block_rmse)

    return rf_random_mae, rf_random_rmse, rf_block_mae, rf_block_rmse, xg_random_mae, xg_random_rmse, xg_block_mae, xg_block_rmse, lg_random_mae, lg_random_rmse, lg_block_mae, lg_block_rmse, ens_random_mae, ens_random_rmse, ens_block_mae, ens_block_rmse

# def train_unet(train_df, epochs=10):
#     """
#     Train U-Net on the given dataset.
    
#     Args:
#         train_df: DataFrame containing training data.
#         epochs: Number of epochs to train for.
    
#     Returns:
#         Trained U-Net model.
#     """
#     X_train, Y_train = preprocess_unet_data(train_df)
    
#     model = build_unet(input_shape=(64, 64, 3))
#     model.fit(X_train, np.expand_dims(Y_train, axis=0), epochs=epochs, verbose=1)
    
#     return model
def train_unet(train_df, epochs=20):
    """
    Train the improved U-Net model.
    
    Args:
        train_df: DataFrame containing training data.
        epochs: Number of epochs to train.
    
    Returns:
        Trained U-Net model.
    """
    # Normalize signal strength to range [-1, 1]
    train_df["rssi"] = (train_df["rssi"] - train_df["rssi"].mean()) / train_df["rssi"].std()
    

    X_train, Y_train = preprocess_unet_data(train_df)
    
    model = build_unet(input_shape=(64, 64, 3))

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.fit(X_train, np.expand_dims(Y_train, axis=0), 
              epochs=epochs, verbose=1, callbacks=[early_stopping])

    return model

def evaluate_unet(model, test_df):
    """
    Evaluate the improved U-Net model on test data.
    
    Args:
        model: Trained U-Net model.
        test_df: DataFrame containing test data.
    
    Returns:
        MAE and RMSE.
    """
    test_df["rssi"] = (test_df["rssi"] - test_df["rssi"].mean()) / test_df["rssi"].std()
    X_test, Y_test = preprocess_unet_data(test_df)
    predictions = model.predict(X_test)[0, :, :, 0]  # Extract 2D output

    mae = np.mean(np.abs(Y_test - predictions))
    rmse = np.sqrt(np.mean((Y_test - predictions) ** 2))

    return mae, rmse


# def do_splatformer(test_size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0, epochs=20):
#     """
#     Executes the Gaussian Splatting + Transformer method on the dataset.
#     """

#     print("Processing Splatformer Method for Test Size:", test_size)

#     # Required feature columns
#     cols_needed = [
#         "gps.lat", "gps.lon", "altitudeAMSL", 
#         "localPosition.x", "localPosition.y", "localPosition.z",
#         "rsrp", "rsrq", "rssi", "sinr"
#     ]

#     # filter out only relevant columns
#     this_train_random = this_train_random[cols_needed].copy()
#     this_test_random = this_test_random[cols_needed].copy()
#     this_train_block = this_train_block[cols_needed].copy()
#     this_test_block = this_test_block[cols_needed].copy()

#     # cluster
#     print("Clustering training data...")
#     clustered_train_random = adaptively_cluster_points(this_train_random)
#     clustered_train_block = adaptively_cluster_points(this_train_block)

#     # ghenerate Gaussians for each cluster
#     print("Generating Gaussian Representations...")
    
#     train_random_gaussians = []
#     for cluster_id in clustered_train_random['cluster'].unique():
#         if cluster_id == -1: continue  # Skip noise points
#         cluster_data = clustered_train_random[clustered_train_random['cluster'] == cluster_id]
#         train_random_gaussians.append(create_gaussian_representation(cluster_data))

#     train_block_gaussians = []
#     for cluster_id in clustered_train_block['cluster'].unique():
#         if cluster_id == -1: continue
#         cluster_data = clustered_train_block[clustered_train_block['cluster'] == cluster_id]
#         train_block_gaussians.append(create_gaussian_representation(cluster_data))

#     # convert list of Gaussians into a single DataFrame
#     train_random_gaussians = pd.concat(train_random_gaussians, ignore_index=True)
#     train_block_gaussians = pd.concat(train_block_gaussians, ignore_index=True)

#     # debugging
#     print(f"Number of clusters in train_random: {clustered_train_random['cluster'].nunique()}")
#     print(f"Number of clusters in train_block: {clustered_train_block['cluster'].nunique()}")

#     #Convert Gaussian Data into Features
#     print("Converting Gaussian Features to DataFrames...")
#     train_random_features = generate_gaussian_features(pd.DataFrame(train_random_gaussians), tower_location=tower_position)
#     train_block_features = generate_gaussian_features(pd.DataFrame(train_block_gaussians), tower_location=tower_position)

#     # train Transformer Model
#     print("Training Transformer on Random Split...")
#     splatformer_random_model = train_gaussian_transformer(train_random_features, this_train_random, epochs=epochs)

#     print("Training Transformer on Block Split...")
#     splatformer_block_model = train_gaussian_transformer(train_block_features, this_train_block, epochs=epochs)

#     # Test Set 
#     test_random_gaussians = create_gaussian_representation(this_test_random)
#     test_block_gaussians = create_gaussian_representation(this_test_block)

#     test_random_features = generate_gaussian_features(pd.DataFrame(test_random_gaussians), tower_location=tower_position)
#     test_block_features = generate_gaussian_features(pd.DataFrame(test_block_gaussians), tower_location=tower_position)

#     # Evaluate model on test set
#     splatformer_rand_mae = evaluate_gaussian_transformer(splatformer_random_model, this_test_random, test_random_features)
#     splatformer_block_mae = evaluate_gaussian_transformer(splatformer_block_model, this_test_block, test_block_features)

#     return splatformer_rand_mae, splatformer_block_mae
def do_splatformer(test_size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0, epochs=20):
    """
    Executes Gaussian Splatting + Transformer method.
    """

    print("Processing Splatformer Method for Test Size:", test_size)

    cols_needed = [
        "gps.lat", "gps.lon", "altitudeAMSL", 
        "localPosition.x", "localPosition.y", "localPosition.z",
        "rsrp", "rsrq", "rssi", "sinr"
    ]



    this_train_random = this_train_random[cols_needed].copy()
    this_test_random = this_test_random[cols_needed].copy()
    this_train_block = this_train_block[cols_needed].copy()
    this_test_block = this_test_block[cols_needed].copy()

    #cluster Training Data
    print("Clustering training data...")
    clustered_train_random = cluster_and_assign_means(this_train_random)
    clustered_train_block = cluster_and_assign_means(this_train_block)

    # create Gaussian Representations per Cluster
    print("Generating Gaussian Representations...")
    train_random_gaussians = create_gaussians_for_clusters(clustered_train_random)
    train_block_gaussians = create_gaussians_for_clusters(clustered_train_block)
    # plot_clusters_with_gaussians(train_random_gaussians)

    # Convert Gaussian Data into Features
    print("Converting Gaussian Features to DataFrames...")
    # train_random_features = generate_gaussian_features(train_random_gaussians, tower_location=tower_position)
    # train_block_features = generate_gaussian_features(train_block_gaussians, tower_location=tower_position)

    # train Transformer Model
    # dist_to_tower = compute_distance_to_tower(data, tower_location).values.reshape(-1, 1)
    # azimuth, elevation = compute_tower_direction(data, tower_location)
    # sinr_weighted_rssi = compute_sinr_weighted_rssi(data).values.reshape(-1, 1)
    # print("Training Transformer on Random Split...")
    # train_random_gaussians = np.hstack([train_random_gaussians
    #     dist_to_tower,  # Standard features
    #     sinr_weighted_rssi,       # New feature
    #     dist_to_tower,            # New feature
    #     azimuth, elevation,       # New directional features
    #     covariance_features       # Covariance matrix
    # ])
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    splatformer_random_model = train_gaussian_transformer(train_random_gaussians, this_train_random, epochs=epochs)

    print("Training Transformer on Block Split...")
    splatformer_block_model = train_gaussian_transformer(train_block_gaussians, this_train_block, epochs=epochs)

    #Test Set (No Clustering, Just Evaluate)
    test_random_gaussians = create_gaussian_representation(this_test_random)
    test_block_gaussians = create_gaussian_representation(this_test_block)

    test_random_features = new_generate_gaussian_features(test_random_gaussians, tower_location=tower_position)
    test_block_features = new_generate_gaussian_features(test_block_gaussians, tower_location=tower_position)


    # Evaluate model on test set
    splatformer_rand_mae = evaluate_gaussian_transformer(splatformer_random_model, this_test_random, test_random_features)
    splatformer_block_mae = evaluate_gaussian_transformer(splatformer_block_model, this_test_block, test_block_features)

    return splatformer_rand_mae, splatformer_block_mae


def evaluation_instance(file_name="placeholder"):

    # MAKE THE MASTER SPLITS

    # train_random, test_random, train_block, test_block = perform_splits(filename="CONF/cleaned_spiral.csv",  this_test_size=0.2, this_n_clusters = 10, this_test_fraction = 0.2)

    # rf_random_mae, rf_random_rmse, rf_block_mae, rf_block_rmse, xg_random_mae, xg_random_rmse, xg_block_mae, xg_block_rmse, lg_random_mae, lg_random_rmse, lg_block_mae, lg_block_rmse, ens_random_mae, ens_random_rmse, ens_block_mae, ens_block_rmse = do_Ensemble(0.2, train_random, test_random, train_block, test_block)


    # Initialize error lists for Random Split
    idw_mae, idw_rmse = [], []
    kriging_mae, kriging_rmse = [], []
    rf_mae, rf_rmse = [], []
    xg_mae, xg_rmse = [], []
    lg_mae, lg_rmse = [], []
    ens_mae, ens_rmse = [], []

    # Initialize error lists for Block Split
    idw_mae_b, idw_rmse_b = [], []
    kriging_mae_b, kriging_rmse_b = [], []
    rf_mae_b, rf_rmse_b = [], []
    xg_mae_b, xg_rmse_b = [], []
    lg_mae_b, lg_rmse_b = [], []
    ens_mae_b, ens_rmse_b = [], []

    # Define test sizes
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Perform splits and collect errors
    for test_size in test_sizes:
        print(f"Processing test size: {test_size}")

        # Perform the dataset splits
        train_random, test_random, train_block, test_block = perform_splits(
            filename=file_name,
            this_test_size=test_size, 
            this_n_clusters=10, 
            this_test_fraction=test_size
        )

        # IDW
        rand_mae, rand_rmse, block_mae, block_rmse = do_IDW(test_size, train_random, test_random, train_block, test_block)
        idw_mae.append(rand_mae)
        idw_rmse.append(rand_rmse)
        idw_mae_b.append(block_mae)
        idw_rmse_b.append(block_rmse)

        # Kriging
        rand_mae, rand_rmse, block_mae, block_rmse = do_Krig(test_size, train_random, test_random, train_block, test_block)
        kriging_mae.append(rand_mae)
        kriging_rmse.append(rand_rmse)
        kriging_mae_b.append(block_mae)
        kriging_rmse_b.append(block_rmse)

        # Ensemble Models (Random Forest, XGBoost, LightGBM, Ensemble)
        rf_rand_mae, rf_rand_rmse, rf_block_mae, rf_block_rmse, \
        xg_rand_mae, xg_rand_rmse, xg_block_mae, xg_block_rmse, \
        lg_rand_mae, lg_rand_rmse, lg_block_mae, lg_block_rmse, \
        ens_rand_mae, ens_rand_rmse, ens_block_mae, ens_block_rmse = do_Ensemble(
            test_size, train_random, test_random, train_block, test_block
        )

        rf_mae.append(rf_rand_mae)
        rf_rmse.append(rf_rand_rmse)
        rf_mae_b.append(rf_block_mae)
        rf_rmse_b.append(rf_block_rmse)

        xg_mae.append(xg_rand_mae)
        xg_rmse.append(xg_rand_rmse)
        xg_mae_b.append(xg_block_mae)
        xg_rmse_b.append(xg_block_rmse)

        lg_mae.append(lg_rand_mae)
        lg_rmse.append(lg_rand_rmse)
        lg_mae_b.append(lg_block_mae)
        lg_rmse_b.append(lg_block_rmse)

        ens_mae.append(ens_rand_mae)
        ens_rmse.append(ens_rand_rmse)
        ens_mae_b.append(ens_block_mae)
        ens_rmse_b.append(ens_block_rmse)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Random Split results
    axes[0].plot(test_sizes, idw_mae, label="IDW MAE", color="blue", marker="o")
    axes[0].plot(test_sizes, idw_rmse, label="IDW RMSE", color="lightblue", linestyle="--", marker="o")
    axes[0].plot(test_sizes, kriging_mae, label="Kriging MAE", color="red", marker="s")
    axes[0].plot(test_sizes, kriging_rmse, label="Kriging RMSE", color="pink", linestyle="--", marker="s")
    axes[0].plot(test_sizes, rf_mae, label="RF MAE", color="green", marker="^")
    axes[0].plot(test_sizes, rf_rmse, label="RF RMSE", color="lightgreen", linestyle="--", marker="^")
    axes[0].plot(test_sizes, xg_mae, label="XGBoost MAE", color="purple", marker="D")
    axes[0].plot(test_sizes, xg_rmse, label="XGBoost RMSE", color="violet", linestyle="--", marker="D")
    axes[0].plot(test_sizes, lg_mae, label="LightGBM MAE", color="orange", marker="X")
    axes[0].plot(test_sizes, lg_rmse, label="LightGBM RMSE", color="gold", linestyle="--", marker="X")
    axes[0].plot(test_sizes, ens_mae, label="Ensemble MAE", color="black", marker="P")
    axes[0].plot(test_sizes, ens_rmse, label="Ensemble RMSE", color="gray", linestyle="--", marker="P")

    # Customize Random Split plot
    axes[0].set_xlabel("Test Size", fontsize=12)
    axes[0].set_ylabel("Error Metric (MAE / RMSE)", fontsize=12)
    axes[0].set_title("Performance of Methods (Random Split)", fontsize=14)
    axes[0].legend()

    # Plot Block Split results
    axes[1].plot(test_sizes, idw_mae_b, label="IDW MAE", color="blue", marker="o")
    axes[1].plot(test_sizes, idw_rmse_b, label="IDW RMSE", color="lightblue", linestyle="--", marker="o")
    axes[1].plot(test_sizes, kriging_mae_b, label="Kriging MAE", color="red", marker="s")
    axes[1].plot(test_sizes, kriging_rmse_b, label="Kriging RMSE", color="pink", linestyle="--", marker="s")
    axes[1].plot(test_sizes, rf_mae_b, label="RF MAE", color="green", marker="^")
    axes[1].plot(test_sizes, rf_rmse_b, label="RF RMSE", color="lightgreen", linestyle="--", marker="^")
    axes[1].plot(test_sizes, xg_mae_b, label="XGBoost MAE", color="purple", marker="D")
    axes[1].plot(test_sizes, xg_rmse_b, label="XGBoost RMSE", color="violet", linestyle="--", marker="D")
    axes[1].plot(test_sizes, lg_mae_b, label="LightGBM MAE", color="orange", marker="X")
    axes[1].plot(test_sizes, lg_rmse_b, label="LightGBM RMSE", color="gold", linestyle="--", marker="X")
    axes[1].plot(test_sizes, ens_mae_b, label="Ensemble MAE", color="black", marker="P")
    axes[1].plot(test_sizes, ens_rmse_b, label="Ensemble RMSE", color="gray", linestyle="--", marker="P")

    # Customize Block Split plot
    axes[1].set_xlabel("Test Size", fontsize=12)
    axes[1].set_ylabel("Error Metric (MAE / RMSE)", fontsize=12)
    axes[1].set_title("Performance of Methods (Block Split)", fontsize=14)
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation_sequence(file_name="", runs=20):
    # Define test sizes
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Initialize dictionaries to store errors
    errors = {method: {"random": [], "block": []} for method in ["IDW", "Kriging", "RF", "XGBoost", "LightGBM", "Ensemble"]}

    for test_size in test_sizes:
        print(f"Processing test size: {test_size}")

        # Run multiple trials
        for _ in range(runs):
            train_random, test_random, train_block, test_block = perform_splits(
                filename=file_name,
                this_test_size=test_size, 
                this_n_clusters=10, 
                this_test_fraction=test_size
            )

            # IDW
            rand_mae, rand_rmse, block_mae, block_rmse = do_IDW(test_size, train_random, test_random, train_block, test_block)
            errors["IDW"]["random"].append(rand_mae)
            errors["IDW"]["block"].append(block_mae)

            # Kriging
            rand_mae, rand_rmse, block_mae, block_rmse = do_Krig(test_size, train_random, test_random, train_block, test_block)
            errors["Kriging"]["random"].append(rand_mae)
            errors["Kriging"]["block"].append(block_mae)

            # Ensemble Models (RF, XGBoost, LightGBM, Ensemble)
            rf_rand_mae, rf_rand_rmse, rf_block_mae, rf_block_rmse, \
            xg_rand_mae, xg_rand_rmse, xg_block_mae, xg_block_rmse, \
            lg_rand_mae, lg_rand_rmse, lg_block_mae, lg_block_rmse, \
            ens_rand_mae, ens_rand_rmse, ens_block_mae, ens_block_rmse = do_Ensemble(
                test_size, train_random, test_random, train_block, test_block
            )

            errors["RF"]["random"].append(rf_rand_mae)
            errors["RF"]["block"].append(rf_block_mae)
            errors["XGBoost"]["random"].append(xg_rand_mae)
            errors["XGBoost"]["block"].append(xg_block_mae)
            errors["LightGBM"]["random"].append(lg_rand_mae)
            errors["LightGBM"]["block"].append(lg_block_mae)
            errors["Ensemble"]["random"].append(ens_rand_mae)
            errors["Ensemble"]["block"].append(ens_block_mae)

    # Compute summary statistics
    summary_stats = {
        method: {
            "Random Mean": np.mean(errors[method]["random"]),
            "Random Min": np.min(errors[method]["random"]),
            "Random Max": np.max(errors[method]["random"]),
            "Random Median": np.median(errors[method]["random"]),
            "Block Mean": np.mean(errors[method]["block"]),
            "Block Min": np.min(errors[method]["block"]),
            "Block Max": np.max(errors[method]["block"]),
            "Block Median": np.median(errors[method]["block"]),
        }
        for method in errors
    }

    # Convert to DataFrame for visualization
    import pandas as pd
    df_summary = pd.DataFrame(summary_stats).T  # Transpose for easier plotting

    # Bar Chart for Mean MAE
    plt.figure(figsize=(12, 6))
    df_summary[['Random Mean', 'Block Mean']].plot(kind='bar', colormap='viridis')
    plt.title("Average MAE for Random vs Block")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.xticks(rotation=45)
    plt.show()

    # Box plots for error distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[errors["RF"]["random"], errors["RF"]["block"]])
    plt.xticks([0, 1], ["Random RF", "Block RF"])
    plt.title("Error Distribution (RF)")
    plt.show()

    return df_summary, errors




def evaluation_seq_2(file_name="", runs=20, epochs=20):
    # test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    methods = ["IDW", "Kriging", "Ensemble", "U-Net", "Splatformer"] 

    # methods = ["IDW", "Kriging", "RF", "XGBoost", "LightGBM", "Ensemble", "U-Net", "Splatformer"] 

    # Data storage for visualization
    error_data_random = []
    error_data_block = []
    tested_sizes = []

    for test_size in test_sizes:
        print(f"Processed test sizes:{tested_sizes} of {test_sizes}")

        for _ in range(runs):
            train_random, test_random, train_block, test_block, scaler = perform_splits(
                filename="CONF/cleaned_spiral.csv",
                this_test_size=test_size, 
                this_n_clusters=20, 
                this_test_fraction=test_size
            )

            # IDW
            rand_mae, _, block_mae, _ = do_IDW(test_size, train_random, test_random, train_block, test_block)
            error_data_random.append(["IDW", test_size, rand_mae])
            error_data_block.append(["IDW", test_size, block_mae])

            # Kriging
            rand_mae, _, block_mae, _ = do_Krig(test_size, train_random, test_random, train_block, test_block)
            error_data_random.append(["Kriging", test_size, rand_mae])
            error_data_block.append(["Kriging", test_size, block_mae])

            # Ensemble Models
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(
                test_size, train_random, test_random, train_block, test_block
            )

            # error_data_random.append(["RF", test_size, rf_rand_mae])
            # error_data_random.append(["XGBoost", test_size, xg_rand_mae])
            # error_data_random.append(["LightGBM", test_size, lg_rand_mae])
            error_data_random.append(["Ensemble", test_size, ens_rand_mae])

            # error_data_block.append(["RF", test_size, rf_block_mae])
            # error_data_block.append(["XGBoost", test_size, xg_block_mae])
            # error_data_block.append(["LightGBM", test_size, lg_block_mae])
            error_data_block.append(["Ensemble", test_size, ens_block_mae])

            # U-Net CNNN
            unet_model = train_unet(train_random, epochs = epochs) #random spli
            unet_rand_mae, _ = evaluate_unet(unet_model, test_random)  # Evaluate on random test set
            unet_model = train_unet(train_random, epochs = epochs) #block split
            unet_block_mae, _ = evaluate_unet(unet_model, test_block)  # Evaluate on block test set

            error_data_random.append(["U-Net", test_size, unet_rand_mae])
            error_data_block.append(["U-Net", test_size, unet_block_mae])


            # print(train_random)
            # Gaussian Splatting + Transformer
            splatformer_rand_mae, splatformer_block_mae = do_splatformer(
                test_size, train_random, test_random, train_block, test_block, epochs=epochs
            )

            error_data_random.append(["Splatformer", test_size, splatformer_rand_mae])
            error_data_block.append(["Splatformer", test_size, splatformer_block_mae])

            if test_size >= 0.6: 
                # Generate Heatmaps
                X_test, Y_test = preprocess_unet_data(test_random)  # Convert test data into a grid
                predictions = unet_model.predict(X_test)[0, :, :, 0]  # Extract predicted signal values

                plot_unet_heatmap(test_random, predictions, title=f"U-Net Heatmap (Random Split, Test Size {test_size})")

                X_test, Y_test = preprocess_unet_data(test_block)
                predictions = unet_model.predict(X_test)[0, :, :, 0]

                plot_unet_heatmap(test_block, predictions, title=f"U-Net Heatmap (Block Split, Test Size {test_size})")

                sns.histplot(train_random["rssi"], kde=True, color="blue", label="Train")
                sns.histplot(test_random["rssi"], kde=True, color="red", label="Test", alpha=0.6)
                sns.histplot(train_block["rssi"], kde=True, color="blue", label="B Train")
                sns.histplot(test_block["rssi"], kde=True, color="red", label="B Test", alpha=0.6)
                plt.legend()
                plt.show()

            error_data_random.append(["U-Net", test_size, unet_rand_mae])
            error_data_block.append(["U-Net", test_size, unet_block_mae])
        
        tested_sizes.append(test_size)

    # Convert to DataFrame for easier plotting
    df_random = pd.DataFrame(error_data_random, columns=["Method", "Test Size", "MAE"])
    df_block = pd.DataFrame(error_data_block, columns=["Method", "Test Size", "MAE"])

    print("Random DataFrame:")
    print(df_random)
    print(df_random.dtypes)

    print("Block DataFrame:")
    print(df_block)
    print(df_block.dtypes)

    df_random['Method'] = df_random['Method'].astype('category')
    df_random['Test Size'] = df_random['Test Size'].astype('category')

    df_block['Method'] = df_block['Method'].astype('category')
    df_block['Test Size'] = df_block['Test Size'].astype('category')

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"CSV_OUT/single_ds_random_{timestamp}.csv"
    df_random.to_csv(filename, index=False) 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"CSV_OUT/single_ds_block_{timestamp}.csv"
    df_block.to_csv(filename, index=False) 

    def plot_violin(df, title):
        # Ensure that columns are treated as categorical
        df['Method'] = df['Method'].astype('category')
        df['Test Size'] = df['Test Size'].astype('category')

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="Test Size", y="MAE", hue="Method", split=True, palette="Set2", inner="quartile")
        plt.title(title, fontsize=14)
        plt.xlabel("Test Size", fontsize=12)
        plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.show()
        
   
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
            "U-Net": "^", "Splatformer": "X"
        }
        linestyles = {
            "IDW": "--", "Kriging": "-.", "Ensemble": ":", 
            "U-Net": (0, (3, 1, 1, 1)), "Splatformer": "-"  # Splatformer always solid
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
        filename = f"singlelayer_performance_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved as {filename}")

        plt.show()


    # Example Usage
    plot_results(df_random, f"{file_name} Single-Layer Perf. (Random)")
    plot_results(df_block, f"{file_name}Single-Layer Perf. (Block)")

    return df_random, df_block

def visualize_cross_dataset_predictions(train_df, test_df, predictions_dict):
    """
    Visualizes how well each method predicts test dataset values using a 3D plot.

    Args:
        train_df (pd.DataFrame): Training dataset (Lat, Lon, Alt, Signal).
        test_df (pd.DataFrame): Test dataset (Lat, Lon, Alt, Signal).
        predictions_dict (dict): Dictionary containing method-wise predicted values for the test set.
                                Example: {"IDW": predicted_values, "Kriging": predicted_values, ...}
    """

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract test set coordinates
    x_test, y_test, z_test = test_df["gps.lat"], test_df["gps.lon"], test_df["altitudeAMSL"]
    
    # Extract training set coordinates
    x_train, y_train, z_train = train_df["gps.lat"], train_df["gps.lon"], train_df["altitudeAMSL"]

    # Plot training points (as small blue dots)
    ax.scatter(x_train, y_train, z_train, color="blue", alpha=0.5, label="Training Data", s=10)

    # Iterate through each method's predictions
    for method, predicted_values in predictions_dict.items():
        # Compute absolute error
        abs_error = np.abs(predicted_values - test_df["rssi"].values)

        # Normalize errors for color mapping
        norm_error = (abs_error - abs_error.min()) / (abs_error.max() - abs_error.min())

        # Scatter plot of test points, colored by error
        scatter = ax.scatter(x_test, y_test, z_test, c=norm_error, cmap="coolwarm", label=f"{method} Prediction", s=40)

        # Draw lines between predicted and actual values
        for i in range(len(x_test)):
            ax.plot([x_test.iloc[i], x_test.iloc[i]], 
                    [y_test.iloc[i], y_test.iloc[i]], 
                    [z_test.iloc[i], z_test.iloc[i] - (predicted_values[i] - test_df["rssi"].iloc[i])], 
                    linestyle="dotted", color="black", alpha=0.5)

    # Add colorbar for error representation
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Prediction Error (dB)")

    # Labels and title
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Cross-Dataset Prediction Accuracy in 3D Space")
    ax.legend()

    plt.show()


def evaluate_cross_dataset(train_file, test_file, runs=5, epochs=20):
    """
    Evaluates model performance by training on one dataset and testing on another dataset.

    Args:
        train_file (str): CSV path for training dataset.
        test_file (str): CSV path for testing dataset.
        runs (int): Number of runs for averaging results.

    Returns:
        DataFrame with evaluation results.
    """

    methods = ["IDW", "Kriging", "Ensemble", "UNET CNN", "Splatformer"]
    error_data_random, error_data_block = [], []
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    for test_size in test_sizes:
        print(f"\n🔹 Processing test size: {test_size}")

        for run in range(runs):
            print(f"Run {run+1}/{runs}: Training on {train_file}, Testing on {test_file}")

            # Load datasets
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            tower_location = tower_position

            # compute the engineered features
            # train_df['distance_to_tower'] = compute_distance_to_tower(train_df, tower_location)
            # test_df['distance_to_tower'] = compute_distance_to_tower(test_df, tower_location)
            # train_df['tower_direction'] = compute_tower_direction(train_df, tower_location)
            # test_df['tower_direction'] = compute_tower_direction(test_df, tower_location)
            # train_df['sinr_weighted_rssi'] = compute_sinr_weighted_rssi(train_df)
            # test_df['sinr_weighted_rssi'] = compute_sinr_weighted_rssi(test_df)

            


            # Split only the test dataset
            _, test_random, _, test_block, _ = perform_splits(
                filename=test_file,  
                this_test_size=test_size, 
                this_n_clusters=10, 
                this_test_fraction=test_size
            )

            # IDW
            rand_mae, _, block_mae, _ = do_IDW(test_size, train_df, test_random, train_df, test_block)
            error_data_random.append(["IDW", test_size, rand_mae])
            error_data_block.append(["IDW", test_size, block_mae])

            # Kriging
            rand_mae, _, block_mae, _ = do_Krig(test_size, train_df, test_random, train_df, test_block)
            error_data_random.append(["Kriging", test_size, rand_mae])
            error_data_block.append(["Kriging", test_size, block_mae])

            # Ensemble Models
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(test_size, train_df, test_random, train_df, test_block)

            # U-Net CNN
            unet_model = train_unet(train_df, epochs=epochs)  
            unet_rand_mae, _ = evaluate_unet(unet_model, test_random)  
            unet_block_mae, _ = evaluate_unet(unet_model, test_block)

            error_data_random.append(["U-Net", test_size, unet_rand_mae])
            error_data_block.append(["U-Net", test_size, unet_block_mae])

            # Splatformer (Transformer + Gaussian Splatting)
            splatformer_rand_mae, splatformer_block_mae = do_splatformer(
                test_size, train_df, test_random, train_df, test_block, epochs=epochs
            )

            # Store results
            error_data_random.extend([
                ["Ensemble", test_size, ens_rand_mae],
                ["U-Net", test_size, unet_rand_mae], ["Splatformer", test_size, splatformer_rand_mae]
            ])
            error_data_block.extend([
                ["Ensemble", test_size, ens_block_mae],
                ["U-Net", test_size, unet_block_mae], ["Splatformer", test_size, splatformer_block_mae]
            ])
            # error_data_block.extend([
            #     ["RF", test_size, rf_block_mae], ["XGBoost", test_size, xg_block_mae],
            #     ["LightGBM", test_size, lg_block_mae], ["Ensemble", test_size, ens_block_mae],
            #     ["U-Net", test_size, unet_block_mae], ["Splatformer", test_size, splatformer_block_mae]
            # ])

    # Convert results to DataFrame
    df_random = pd.DataFrame(error_data_random, columns=["Method", "Test Size", "MAE"])
    df_block = pd.DataFrame(error_data_block, columns=["Method", "Test Size", "MAE"])

     # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"CSV_OUT/cross_ds_random_{timestamp}.csv"
    df_random.to_csv(filename, index=False) 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"CSV_OUT/cross_ds_block_{timestamp}.csv"
    df_block.to_csv(filename, index=False) 

    
    def plot_violin(df, title):
        """
        Improved violin plot with larger text, print-friendly layout, and proper distribution.

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
            "legend.fontsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.linewidth": 3,
            "lines.markersize": 10
        })

        # Ensure categorical data types
        df['Method'] = df['Method'].astype('category')
        df['Test Size'] = df['Test Size'].astype(float)

        fig, ax = plt.subplots(figsize=(9, 9))  # **Square Figure**
        
        sns.violinplot(
            data=df, x="Test Size", y="MAE", hue="Method",
            palette="Reds", inner="box", density_norm="width", ax=ax
        )

        ax.set_title(f"{title} Violin", fontsize=20)
        ax.set_xlabel("Test Size", fontsize=18)
        ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=18)
        ax.grid(True, linestyle="dotted", linewidth=1.5)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()

        # --- Save figure as PDF ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        filename = f"crosslayer_violin_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved as {filename}")

        plt.show()


    
    def plot_results(df, title):
        """
        Improved visualization with a Y-axis break, placing high MAE values at the top.

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

        # --- Define Breakpoint for the Y-Axis ---
        low_threshold = 6   # Show small MAE values fully
        high_threshold = 75  # Compress values above this

        # Create figure with two subplots sharing x-axis (Flipped order)
        fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(9, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # --- Define Line Styles & Markers ---
        markers = {
            "IDW": "o", "Kriging": "s", "Ensemble": "D",
            "U-Net": "^", "Splatformer": "X"
        }
        linestyles = {
            "IDW": "--", "Kriging": "-.", "Ensemble": ":",
            "U-Net": (0, (3, 1, 1, 1)), "Splatformer": "-"  # Splatformer always solid
        }

        unique_methods = df["Method"].unique()
        
        # --- Plot the Data Twice (Once for Each Axis) ---
        for method in unique_methods:
            method_df = df[df["Method"] == method]

            # Plot upper range (compressed region) FIRST
            sns.lineplot(
                data=method_df, x="Test Size", y="MAE",
                label=method if method in ["IDW", "Kriging", "Ensemble"] else "",  # Avoid duplicate legends
                marker=markers.get(method, "o"), linestyle=linestyles.get(method, "--"),
                markersize=10, linewidth=3, ax=ax2
            )

            # Plot lower range (0 - 2 MAE)
            sns.lineplot(
                data=method_df, x="Test Size", y="MAE",
                label=method if method in ["Splatformer", "U-Net"] else "",  # Avoid duplicate legends
                marker=markers.get(method, "o"), linestyle=linestyles.get(method, "--"),
                markersize=10, linewidth=3, ax=ax1
            )

        # --- Adjust the Y-Axes ---
        ax1.set_ylim(0, low_threshold)  # Show 0-2 range fully
        ax2.set_ylim(high_threshold, df["MAE"].max() + 15)  # Show large values compressed

        # Hide the spines between the two plots
        ax1.spines.top.set_visible(False)
        ax2.spines.bottom.set_visible(False)

        # # Add diagonal break marks
        # d = 0.015
        # kwargs = dict(transform=ax2.transAxes, color='black', clip_on=False)
        # ax2.plot((-d, +d), (-d, +d), **kwargs)
        # ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        # kwargs.update(transform=ax1.transAxes)
        # ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        # ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # --- Labels & Titles ---
        ax1.set_ylabel("", fontsize=18)
        ax2.set_ylabel("MAE", fontsize=18)
        ax1.set_xlabel("Test Size", fontsize=18)
        ax2.set_title(f"{title}", fontsize=20)
        ax1.grid(True, linestyle="dotted", linewidth=1.5)
        ax2.grid(True, linestyle="dotted", linewidth=1.5)

        # Legends
        ax1.legend(loc="upper right", frameon=True)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()

        # --- Save Figure as PDF ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        filename = f"crosslayer_performance_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved as {filename}")

    
        if show_res: plt.show()


    # --- Run the Function ---
    plot_results(df_random, f"{train_file}Cross-layer Perf. (Random)")
    plot_results(df_block, f"{train_file}Cross-layer Perf. (Block)")
                    
    return df_random, df_block
  # Input
    # train_random, test_random, train_block, test_block, scaler = perform_splits(
    #             filename=file_name,
    #             this_test_size=0.3, 
    #             this_n_clusters=20, 
    #             this_test_fraction=0.3
    #         )

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def new_method(file_name="CONF/cleaned_rows.csv", show_res=True):
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
    dataset = pd.read_csv(file_name)
    train = dataset[['gps.lat', 'gps.lon', 'altitudeAMSL', 'localPosition.x','localPosition.y','localPosition.z', 'pressure', 'temperature', 'humidity', 'gas_resistance']]

    # Feature engineering
    # Tower distance
    distances = compute_distance_to_tower(train, tower_location=tower_position)/1000
    train['distance_to_tower'] = distances
    # Tower direction
    _, elevation_angle = compute_tower_direction_local(train)
    azimuths,_ = compute_tower_direction(train, tower_location=tower_position)
    train['azimuth'] = azimuths # add az to input
    train['elevation_angle'] = elevation_angle # add e angle to input
    print(distances, azimuths, elevation_angle)

    # Input features: spatial relationships + absolute positioning
    features = ['distance_to_tower', 'azimuth', 'elevation_angle', 'gps.lat', 'gps.lon', 'altitudeAMSL', 'pressure', 'temperature', 'humidity', 'gas_resistance']
    
    # for testing
    sinr_values = dataset['sinr'].values  # SINR for loss weighting
    true_rsrp = dataset['rsrp'].values  # Actual RSRP values
    true_rsrq = dataset['rsrq'].values  # Actual RSRQ values
    true_sinr = dataset['sinr'].values  # Actual SINR values

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
    true_values = [dataset['rsrp'].values, dataset['rsrq'].values, dataset['sinr'].values]
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


















    #old
    # dataset = pd.read_csv(file_name)
    # train = dataset[['gps.lat', 'gps.lon', 'altitudeAMSL']]
    # print(train)

    # # Feature engineering
    # # Tower distance
    # distances = compute_distance_to_tower(train, tower_location=tower_position)/1000
    # train['distance_to_tower'] = distances
    # # Tower direction
    # azimuths, elevation_angle = compute_tower_direction(train, tower_location=tower_position)
    # train['azimuth'] = azimuths # add az to input
    # train['elevation_angle'] = elevation_angle # add e angle to input
    # print(distances, azimuths, elevation_angle)

    # # sinr_weighted_rssi = compute_sinr_weighted_rssi(data).values.reshape(-1, 1)
    # print(distances, azimuths, elevation_angle)
    # print(train.head(10))
    # train.to_csv('NEW.csv', index=False) # log updated input as csv

    # transformer








if __name__ == '__main__':
    # evaluation_instance(file_name="CONF/cleaned_rows.csv")
    # evaluation_sequence(file_name="CONF/cleaned_spiral.csv", runs=20)
    
    # print(evaluation_seq_2(file_name="CONF/cleaned_rows.csv", runs=20, epochs=100))

    # train_block_features  = pd.read_csv("CONF/cleaned_rows.csv")
    # plot_flightpath_combined(train_block_features, tower_location=tower_position)
    # plot_sinr_weighted_rssi(train_block_features)

    # df_comparison = evaluate_cross_dataset("CONF/cleaned_rows.csv", "CONF/cleaned_spiral.csv", runs=5, epochs=10)# train on 20m rows, predict 15m spiral
    # df_comparison = evaluate_cross_dataset("CONF/cleaned_spiral.csv", "CONF/cleaned_rows.csv", runs=5, epochs=10)#rain on 15m spiral, predict 20m rows
    # print(df_comparison)

    # Figure 4: Rows and SPiral single layer perfomance
    # print(evaluation_seq_2(file_name="CONF/cleaned_spiral.csv", runs=3, epochs=5))
    # print(evaluation_seq_2(file_name="CONF/cleaned_rows.csv", runs=1, epochs=10))




    #  Conference figures 

    ### Fig. 3. Example test-train split
    # _, _, _, _ = perform_splits(filename="CONF/cleaned_spiral.csv", this_test_size=0.3, this_n_clusters=10, this_test_fraction=0.3, visualise=True)
    
    show_res=True
    new_method("CONF/cleaned_rows.csv")
    
    new_method("CONF/cleaned_spiral2.csv")

    ### Fig. 4. Engineered feature - Distance to Tower, Bearing of Tower, SINR Weighting
    # rows_example_features  = pd.read_csv("CONF/cleaned_rows.csv")
    # spiral_example_features  = pd.read_csv("CONF/cleaned_spiral.csv")
    # plot_flightpath_combined(rows_example_features, tower_location=tower_position)
    # plot_flightpath_combined(spiral_example_features, tower_location=tower_position)
    # plot_sinr_weighted_rssi(rows_example_features)

    ### Figure 5 & 6. Rows and Spiral Single-layer prediction
    # print(evaluation_seq_2(file_name="CONF/cleaned_rows.csv", runs=1, epochs=5))
    # print("EVAL SEQ DONE")
    # print(evaluation_seq_2(file_name="CONF/cleaned_spiral.csv", runs=6, epochs=100))
    # print("EVAL SEQ DONE")
    

    # ### Figure 7 & 8. RSSI layer-to-layer projection
    # df_comparison = evaluate_cross_dataset("CONF/cleaned_rows.csv", "CONF/cleaned_spiral.csv", runs=6, epochs=100) # train on 20m rows, predict 15m spiral, 50 runs over 200 epoch: 4hrs
    # print("EVAL SEQ DONE")
    # show_res = True
    # df_comparison = evaluate_cross_dataset("CONF/cleaned_spiral.csv", "CONF/cleaned_rows.csv", runs=6, epochs=100) #rain on 15m spiral, predict 20m rows

    

    