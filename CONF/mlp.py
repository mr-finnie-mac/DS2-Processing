import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras_tuner as kt

def build_mlp(input_shape, num_layers=2, num_neurons=128, dropout_rate=0.5, lr = 0.01):
    model = tf.keras.models.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Dense(num_neurons, input_shape=(input_shape,), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    # Hidden layers
    for _ in range(num_layers - 1):  # num_layers includes input, so we subtract 1
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())  # Helps with training stability
        model.add(tf.keras.layers.Dropout(dropout_rate))  # Helps with overfitting
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))  # Predicting RSSI
    # from keras.losses import Huber
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model

def run_mlp(X_train, y_train, X_test, y_test, input_dim = 3, epochs=100):
    input_shape = X_train.shape[1]
    mlp_model = build_mlp(input_shape=input_dim, num_layers=2, num_neurons=256, dropout_rate=0.5, lr=0.01)
    mlp_model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = mlp_model.fit(
        X_train, y_train, 
        epochs=epochs, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stop],
        verbose=1
    )

    return mlp_model

def build_mlp_tuner(hp):
    model = tf.keras.models.Sequential()
    
    # Tune number of neurons (64, 128, 256)
    num_neurons = hp.Choice('num_neurons', values=[64, 128, 256])
    
    # Input layer
    model.add(tf.keras.layers.Dense(num_neurons, activation='relu', input_shape=(6,)))
    model.add(BatchNormalization())

    # Tune number of hidden layers (2, 3, or 4)
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)

    for _ in range(num_layers - 1):  # Already have 1 input layer, so add num_layers - 1 more
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
        model.add(BatchNormalization())

        # Tune dropout rate (0.2, 0.3, 0.5)
        dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.5])
        model.add(Dropout(dropout_rate))
    
    # Output layer (predicting RSSI)
    model.add(tf.keras.layers.Dense(1))
    
    # Tune learning rate (0.01, 0.001, 0.0001)
    learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model
