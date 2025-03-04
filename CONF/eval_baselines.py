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
from gaussian import compute_anisotropic_covariance, adaptively_cluster_points, create_gaussian_representation

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
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

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


def do_splatformer(test_size, this_train_random=0, this_test_random=0, this_train_block=0, this_test_block=0):
    """
    Executes the Gaussian Splatting + Transformer method on the dataset.
    """
    
    print("Available columns in train_random before filtering:", this_train_random.columns)

    # 🛠 Define all necessary columns
    cols_needed = ["gps.lat", "gps.lon", "altitudeAMSL", "localPosition.x", "localPosition.y", "localPosition.z",
                   "rsrp", "rsrq", "rssi", "sinr"]
    this_train_random = this_train_random[cols_needed].copy()
    this_test_random = this_test_random[cols_needed].copy()
    
    # Check for missing columns in training and testing data
    missing_train_cols = set(cols_needed) - set(this_train_random.columns)
    missing_test_cols = set(cols_needed) - set(this_test_random.columns)
    
    if missing_train_cols:
        raise KeyError(f"Missing columns in training data: {missing_train_cols}")
    if missing_test_cols:
        raise KeyError(f"Missing columns in testing data: {missing_test_cols}")

    # 🛠 Filter only the required columns
    this_train_random = this_train_random[cols_needed]
    this_test_random = this_test_random[cols_needed]
    this_train_block = this_train_block[cols_needed]
    this_test_block = this_test_block[cols_needed]

    print("Filtered train_random columns:", this_train_random.columns)

    # Continue with the normal pipeline
    train_random_gaussians = create_gaussian_representation(this_train_random)
    test_random_gaussians = create_gaussian_representation(this_test_random)

    train_random_features = generate_gaussian_features(train_random_gaussians)
    test_random_features = generate_gaussian_features(test_random_gaussians)

    # Train the Gaussian Transformer model for random data
    splatformer_random_model = train_gaussian_transformer(features=train_random_features, train_data=this_train_random, test_data=this_test_random)
    
    # Set the model to evaluation mode before evaluation
    splatformer_random_model.eval()  

    # Evaluate the model
    splatformer_rand_mae = evaluate_gaussian_transformer(splatformer_random_model, this_test_random, test_random_features)

    # Process block data
    train_block_gaussians = create_gaussian_representation(this_train_block)
    test_block_gaussians = create_gaussian_representation(this_test_block)

    train_block_features = generate_gaussian_features(train_block_gaussians)
    test_block_features = generate_gaussian_features(test_block_gaussians)

    # Train the Gaussian Transformer model for block data
    splatformer_block_model = train_gaussian_transformer(features=train_block_features, train_data=this_train_block, test_data=this_test_block)
    
    # Set the block model to evaluation mode before evaluation
    splatformer_block_model.eval()  

    # Evaluate the model
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




def evaluation_seq_2(file_name="", runs=20):
    # test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    methods = ["IDW", "Kriging", "Ensemble", "U-Net", "[OURS] Splatformer"] 

    # methods = ["IDW", "Kriging", "RF", "XGBoost", "LightGBM", "Ensemble", "U-Net", "[OURS] Splatformer"] 

    # Data storage for visualization
    error_data_random = []
    error_data_block = []

    for test_size in test_sizes:
        print(f"Processing test size: {test_size}")

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
            unet_model = train_unet(train_random)  # Train on random split
            unet_rand_mae, _ = evaluate_unet(unet_model, test_random)  # Evaluate on random test set
            unet_block_mae, _ = evaluate_unet(unet_model, test_block)  # Evaluate on block test set

            error_data_random.append(["U-Net", test_size, unet_rand_mae])
            error_data_block.append(["U-Net", test_size, unet_block_mae])


            # print(train_random)
            # Gaussian Splatting + Transformer
            splatformer_rand_mae, splatformer_block_mae = do_splatformer(
                test_size, train_random, test_random, train_block, test_block
            )

            error_data_random.append(["[OURS] Splatformer", test_size, splatformer_rand_mae])
            error_data_block.append(["[OURS] Splatformer", test_size, splatformer_block_mae])

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


    df_random.to_csv('df_randomout.csv', index=False) 
    df_block.to_csv('df_blockout.csv', index=False) 

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
        Plot both a violin plot and a line graph for MAE performance of different methods.

        Args:
            df: DataFrame containing results (Method, Test Size, MAE)
            title: Title for the plot
        """

        # Ensure categorical data types for better plotting
        df['Method'] = df['Method'].astype('category')
        df['Test Size'] = df['Test Size'].astype(float)  # Convert to float for line plot

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Violin Plot ---
        sns.violinplot(data=df, x="Test Size", y="MAE", hue="Method", split=True, palette="Set2", inner="quartile", ax=axes[0])
        axes[0].set_title(f"{title} - Violin Plot", fontsize=14)
        axes[0].set_xlabel("Test Size", fontsize=12)
        axes[0].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[0].grid(True)

        # --- Line Plot ---
        sns.lineplot(data=df, x="Test Size", y="MAE", hue="Method", marker="o", palette="Set1", ax=axes[1])
        axes[1].set_title(f"{title} - Line Plot", fontsize=14)
        axes[1].set_xlabel("Test Size", fontsize=12)
        axes[1].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        # Now run your plotting
        # plot_violin(df_random, "Performance of Methods (Random Split)")
        # plot_violin(df_block, "Performance of Methods (Block Split)")


    plot_results(df_random, "Performance of Methods (Random Split)")
    plot_results(df_block, "Performance of Methods (Block Split)")

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


def evaluate_cross_dataset(train_file, test_file, runs=5):
    """
    Evaluate model performance when training on one dataset (lower height)
    and testing on another dataset (higher height).

    Args:
        train_file (str): CSV file path for training dataset.
        test_file (str): CSV file path for testing dataset.
        runs (int): Number of runs to average performance.

    Returns:
        DataFrame with evaluation results.
    """
    
    methods = ["IDW", "Kriging", "Ensemble", "UNET CNN", "Splatformer"]
    error_data_random = []
    error_data_block = []
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    for test_size in test_sizes:
        print(f"Processing test size: {test_size}")

        for _ in range(runs):
            print(f"Running Cross-Dataset Evaluation (Train: {train_file}, Test: {test_file})")

            # Load datasets
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            # Subset test dataset at its unique height
            # test_height = test_df["localPosition.z"].unique()[0]
            # test_subset = test_df[test_df["localPosition.z"] == test_height]

            # Here we split up the test file only, we can use the same fucntion but only use the test split.
            _, test_random, _ , test_block, scaler = perform_splits(
                filename=test_file,  # Pass DataFrame directly
                this_test_size=test_size, 
                this_n_clusters=10, 
                this_test_fraction=test_size
            )

            random_layer_test = test_random
            block_layer_test = test_block

            # IDW
            rand_mae, _, block_mae, _ = do_IDW(test_size, train_df, random_layer_test, train_df, block_layer_test)
            error_data_random.append(["IDW", test_size, rand_mae])
            error_data_block.append(["IDW", test_size, block_mae])

            # Kriging
            rand_mae, _, block_mae, _ = do_Krig(test_size, train_df, random_layer_test, train_df, block_layer_test)
            error_data_random.append(["Kriging", test_size, rand_mae])
            error_data_block.append(["Kriging", test_size, block_mae])

            # Ensemble Models
            rf_rand_mae, _, rf_block_mae, _, \
            xg_rand_mae, _, xg_block_mae, _, \
            lg_rand_mae, _, lg_block_mae, _, \
            ens_rand_mae, _, ens_block_mae, _ = do_Ensemble(test_size, train_df, random_layer_test, train_df, block_layer_test)
            
            # U-Net CNNN
            unet_model = train_unet(train_df)  # Train on random split
            unet_rand_mae, _ = evaluate_unet(unet_model, random_layer_test)  # Evaluate on random test set
            unet_block_mae, _ = evaluate_unet(unet_model, block_layer_test)  # Evaluate on block test set

            # Splatformer
            splatformer_rand_mae, splatformer_block_mae = do_splatformer(
                test_size, train_df, random_layer_test, train_df, block_layer_test
            )

            error_data_random.append(["RF", test_size, rf_rand_mae])
            error_data_block.append(["RF", test_size, rf_block_mae])

            error_data_random.append(["XGBoost", test_size, xg_rand_mae])
            error_data_block.append(["XGBoost", test_size, xg_block_mae])

            error_data_random.append(["LightGBM", test_size, lg_rand_mae])
            error_data_block.append(["LightGBM", test_size, lg_block_mae])

            error_data_random.append(["Ensemble", test_size, ens_rand_mae])
            error_data_block.append(["Ensemble", test_size, ens_block_mae])

            
            error_data_random.append(["U-Net", test_size, unet_rand_mae])
            error_data_block.append(["U-Net", test_size, unet_block_mae])

            error_data_random.append(["Splatformer", test_size, splatformer_rand_mae])
            error_data_block.append(["Splatformer", test_size, splatformer_block_mae])

    # Convert results to DataFrame
    # df_results = pd.DataFrame(error_data, columns=["Method", "Test Height", "MAE"])
    
    # Save to CSV
    # df_results.to_csv("cross_dataset_results.csv", index=False)
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


    df_random.to_csv('layer_df_randomout.csv', index=False) 
    df_block.to_csv('layer_df_blockout.csv', index=False) 

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
        Plot both a violin plot and a line graph for MAE performance of different methods.

        Args:
            df: DataFrame containing results (Method, Test Size, MAE)
            title: Title for the plot
        """

        # Ensure categorical data types for better plotting
        df['Method'] = df['Method'].astype('category')
        df['Test Size'] = df['Test Size'].astype(float)  # Convert to float for line plot

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Violin Plot ---
        sns.violinplot(data=df, x="Test Size", y="MAE", hue="Method", split=True, palette="Set2", inner="quartile", ax=axes[0])
        axes[0].set_title(f"{title} - Violin Plot", fontsize=14)
        axes[0].set_xlabel("Test Size", fontsize=12)
        axes[0].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[0].grid(True)

        # --- Line Plot ---
        sns.lineplot(data=df, x="Test Size", y="MAE", hue="Method", marker="o", palette="Set1", ax=axes[1])
        axes[1].set_title(f"{title} - Line Plot", fontsize=14)
        axes[1].set_xlabel("Test Size", fontsize=12)
        axes[1].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        # Now run your plotting
        # plot_violin(df_random, "Performance of Methods (Random Split)")
        # plot_violin(df_block, "Performance of Methods (Block Split)")


    plot_results(df_random, "Cross-layer projection - Performance of Methods (Random Split)")
    plot_results(df_block, "Cross-layer projection - Performance of Methods (Block Split)")

    return df_random, df_block




if __name__ == '__main__':
    # evaluation_instance(file_name="CONF/cleaned_rows.csv")
    # evaluation_sequence(file_name="CONF/cleaned_spiral.csv", runs=20)
    
    # print(evaluation_seq_2(file_name="CONF/cleaned_rows.csv", runs=2))

    df_comparison = evaluate_cross_dataset("CONF/cleaned_rows.csv", "CONF/cleaned_spiral.csv", runs=2)
    # print(df_comparison)