import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping
import config

def build_unet(input_shape=(64, 64, 3)):
    """
    Builds an improved U-Net model for spatial signal strength prediction.

    Args:
        input_shape: Tuple representing input shape (height, width, channels).
    
    Returns:
        Compiled U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = LeakyReLU(negative_slope=0.1)(c1)
    c1 = Conv2D(64, (3, 3), padding='same')(c1)
    c1 = LeakyReLU(negative_slope=0.1)(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = LeakyReLU(negative_slope=0.1)(c2)
    c2 = Conv2D(128, (3, 3), padding='same')(c2)
    c2 = LeakyReLU(negative_slope=0.1)(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), padding='same')(p2)
    c3 = LeakyReLU(negative_slope=0.1)(c3)
    c3 = Conv2D(256, (3, 3), padding='same')(c3)
    c3 = LeakyReLU(negative_slope=0.1)(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = LeakyReLU(negative_slope=0.1)(c4)
    c4 = Dropout(0.3)(c4)  # Added Dropout
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = LeakyReLU(negative_slope=0.1)(c4)

    # Decoder
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, (3, 3), padding='same')(u5)
    c5 = LeakyReLU(negative_slope=0.1)(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, (3, 3), padding='same')(u6)
    c6 = LeakyReLU(negative_slope=0.1)(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, (3, 3), padding='same')(u7)
    c7 = LeakyReLU(negative_slope=0.1)(c7)
    c7 = BatchNormalization()(c7)

    outputs = Conv2D(1, (1, 1), activation='linear')(c7)  # Output is signal strength

    model = Model(inputs, outputs)
    # model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
    model.compile(optimizer="adam", loss=tf.keras.losses.LogCosh(), metrics=["mae"]) # penalizes large prediction errors less aggressively than MSE, leading to more stable learning
    

    
    return model


import numpy as np
from scipy.interpolate import griddata

def preprocess_unet_data(df, grid_size=(64, 64)):
    """
    Converts dataset into a structured 2D grid for U-Net training.
    
    Args:
        df: Pandas DataFrame containing lat/lon and signal strength data.
        grid_size: Desired grid resolution.
    
    Returns:
        X (input grid), Y (target signal strength grid)
    """
    # Extract necessary fields
    points = df[['gps.lat', 'gps.lon']].values
    values = df['rssi'].values

    # Generate grid
    grid_x, grid_y = np.mgrid[
        df['gps.lat'].min():df['gps.lat'].max():grid_size[0]*1j,
        df['gps.lon'].min():df['gps.lon'].max():grid_size[1]*1j
    ]

    # Interpolate signal strength values onto grid
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Normalize
    grid_z = np.nan_to_num(grid_z)  # Handle NaNs

    # Reshape for CNN input
    X = np.stack([grid_x, grid_y, grid_z], axis=-1)  # 3 channels: lat, lon, signal strength
    X = np.expand_dims(X, axis=0)  # Add batch dimension

    return X, grid_z

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def plot_unet_heatmap(test_df, predictions, title="U-Net Predictions vs Actual"):
    """
    Generates a heatmap comparing actual vs. predicted signal strength and their absolute difference.

    Args:
        test_df: DataFrame containing actual signal strength values.
        predictions: 2D NumPy array of predicted signal strength values.
        title: Title for the heatmap visualization.
    """
    # Extract lat/lon and actual signal strength
    latitudes = test_df["gps.lat"].values
    longitudes = test_df["gps.lon"].values
    actual_values = test_df["rssi"].values  # Ensure this matches your dataset's signal column

    # Get the grid shape from U-Net's predictions
    grid_shape = predictions.shape

    # Create grid for interpolation
    grid_x, grid_y = np.mgrid[
        latitudes.min():latitudes.max():grid_shape[0]*1j,
        longitudes.min():longitudes.max():grid_shape[1]*1j
    ]

    # Interpolate actual values onto the same grid as U-Net predictions
    actual_grid = griddata((latitudes, longitudes), actual_values, (grid_x, grid_y), method='cubic')
    actual_grid = np.nan_to_num(actual_grid)  # Handle NaNs from interpolation

    # Compute absolute error (|Actual - Predicted|)
    abs_error_grid = np.abs(actual_grid - predictions)

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Actual Signal Strength Heatmap (Interpolated)
    sns.heatmap(actual_grid, ax=axes[0], cmap="coolwarm", annot=False)
    axes[0].set_title("Actual Signal Strength (Interpolated)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Predicted Signal Strength Heatmap
    sns.heatmap(predictions, ax=axes[1], cmap="coolwarm", annot=False)
    axes[1].set_title("Predicted Signal Strength (U-Net)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    # Absolute Error Heatmap
    sns.heatmap(abs_error_grid, ax=axes[2], cmap="magma", annot=False)
    axes[2].set_title("Absolute Error (|Actual - Predicted|)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")

    # Adjust layout
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
