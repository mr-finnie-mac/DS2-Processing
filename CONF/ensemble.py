import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from spatial_test_train_split import perform_splits
import config
from sklearn.preprocessing import StandardScaler



# # Load and split dataset using spatial methods
# train_random, test_random, train_block, test_block = perform_splits(filename="CONF/cleaned_spiral.csv")

# # Features and target selection
# position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
# signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]

# Define function to train models and compute ensemble
def train_and_evaluate(train_df, test_df, target_col="rssi", position_cols=0):

    # print("Columns in train_df:", list(train_df.columns))
    # print("position_cols:", position_cols)
    # print("target_col:", target_col)

    X_train, y_train = train_df[position_cols], train_df[target_col]
    X_test, y_test = test_df[position_cols], test_df[target_col]

    # Initialize models with default parameters
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    lgb_params = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "boosting_type": "gbdt",
        "num_leaves": 50,  # Increase complexity
        "max_depth": -1,  # No depth limit
        "learning_rate": 0.05,  # Moderate learning rate
        "min_gain_to_split": 0.001,  # Allow smaller gain splits
        "min_data_in_leaf": 10,  # Reduce minimum leaf size
        "max_bin": 1024,  # More bins for better granularity
        "verbose": -1  # Suppress warnings for cleaner output
    }

    lgb_model = lgb.LGBMRegressor(**lgb_params)

    # Train models
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)

    # Make predictions
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)

    # Simple Averaging Ensemble
    ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3

    # Evaluate performance
    def evaluate(y_true, y_pred):
        

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}

    return {
        "Random Forest": evaluate(y_test, rf_pred),
        "XGBoost": evaluate(y_test, xgb_pred),
        "LightGBM": evaluate(y_test, lgb_pred),
        "Ensemble": evaluate(y_test, ensemble_pred),
    }

# Evaluate on both random and block-based splits
# results_random = train_and_evaluate(train_random, test_random)
# results_block = train_and_evaluate(train_block, test_block)

def perform_ensemble(size=0.2, train_random=0, test_random=0, train_block=0, test_block=0):
    # Load and split dataset using spatial methods
    # train_random, test_random, train_block, test_block = perform_splits(filename="CONF/cleaned_spiral.csv", this_test_size=size)

    # Features and target selection
    position_cols = ["gps.lat", "gps.lon", "localPosition.x", "localPosition.y", "localPosition.z"]
    signal_cols = ["rsrq", "rsrp", "rssi", "sinr"]

    # Evaluate on both random and block-based splits
    results_random = train_and_evaluate(train_random, test_random, target_col="rssi", position_cols=position_cols)
    results_block = train_and_evaluate(train_block, test_block, target_col="rssi", position_cols=position_cols)

    return results_random, results_block


# # Print results
# print("Performance on Random Split:")
# for model, metrics in results_random.items():
#     print(f"{model}: {metrics}")

# print("\nPerformance on Spatial Block Split:")
# for model, metrics in results_block.items():
#     print(f"{model}: {metrics}")
