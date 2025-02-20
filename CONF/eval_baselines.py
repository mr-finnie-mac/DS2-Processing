from sklearn.metrics import mean_absolute_error, mean_squared_error
from IDW import perform_IDW
from krig import perform_Krig
from ensemble import perform_ensemble
import pandas as pd
import numpy as np
from spatial_test_train_split import perform_splits

import matplotlib.pyplot as plt


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
    test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    methods = ["IDW", "Kriging", "RF", "XGBoost", "LightGBM", "Ensemble"]

    # Data storage for visualization
    error_data_random = []
    error_data_block = []

    for test_size in test_sizes:
        print(f"Processing test size: {test_size}")

        for _ in range(runs):
            train_random, test_random, train_block, test_block = perform_splits(
                filename="CONF/cleaned_spiral.csv",
                this_test_size=test_size, 
                this_n_clusters=10, 
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

            error_data_random.append(["RF", test_size, rf_rand_mae])
            error_data_random.append(["XGBoost", test_size, xg_rand_mae])
            error_data_random.append(["LightGBM", test_size, lg_rand_mae])
            error_data_random.append(["Ensemble", test_size, ens_rand_mae])

            error_data_block.append(["RF", test_size, rf_block_mae])
            error_data_block.append(["XGBoost", test_size, xg_block_mae])
            error_data_block.append(["LightGBM", test_size, lg_block_mae])
            error_data_block.append(["Ensemble", test_size, ens_block_mae])

    # Convert to DataFrame for easier plotting
    df_random = pd.DataFrame(error_data_random, columns=["Method", "Test Size", "MAE"])
    df_block = pd.DataFrame(error_data_block, columns=["Method", "Test Size", "MAE"])

    # Plot function using violin plots
    def plot_violin(df, title):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="Test Size", y="MAE", hue="Method", split=True, palette="Set2", inner="quartile")
        plt.title(title, fontsize=14)
        plt.xlabel("Test Size", fontsize=12)
        plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.show()

    # Plot Random Split Results
    plot_violin(df_random, "Performance of Methods (Random Split)")

    # Plot Block Split Results
    plot_violin(df_block, "Performance of Methods (Block Split)")

    return df_random, df_block


if __name__ == '__main__':
    # evaluation_instance(file_name="CONF/cleaned_rows.csv")
    # evaluation_sequence(file_name="CONF/cleaned_spiral.csv", runs=20)
    print(evaluation_seq_2(file_name="CONF/cleaned_rows.csv", runs=20))