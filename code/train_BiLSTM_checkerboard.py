import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Concatenate, TimeDistributed, Conv1D, \
    MaxPooling1D, Flatten, Bidirectional, RepeatVector, Reshape, \
    Dropout, BatchNormalization, LayerNormalization
# from keras.layers import Attention, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
# from keract import get_activations


# In this example we need it because we want to extract all the intermediate output values.
os.environ['KERAS_ATTENTION_DEBUG'] = '1'
from attention import Attention


def normalize_data(train_X, train_y):
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    # X_scaler = RobustScaler()
    # y_scaler = RobustScaler()

    num_data, time_lag, num_features = train_X.shape
    train_X = train_X.reshape(num_data * time_lag, num_features)

    train_X_scaled = X_scaler.fit_transform(train_X)
    train_y_scaled = y_scaler.fit_transform(train_y)

    # Exporting Scalers
    joblib.dump(X_scaler, "../model/LSTM_Scaler/X_scaler.pkl")
    joblib.dump(y_scaler, "../model/LSTM_Scaler/y_scaler.pkl")

    return train_X_scaled, train_y_scaled, X_scaler, y_scaler


def fit_lstm_model(train_X, train_y, n_batch, n_epoch, n_neurons, val=0.05):
    # Define the model
    ts_input = Input((train_X.shape[1], train_X.shape[2]))
    mask_layer = tf.keras.layers.Masking(mask_value=-1, )(ts_input)

    lstm1_1_out = Bidirectional(LSTM(int(n_neurons), activation="tanh", return_sequences=True))(mask_layer)
    lstm1_1_lynorm = LayerNormalization()(lstm1_1_out)
    dropout_1 = Dropout(0.2)(lstm1_1_lynorm)

    lstm1_2_out = Bidirectional(LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True))(dropout_1)
    lstm1_2_lynorm = LayerNormalization()(lstm1_2_out)
    dropout_2 = Dropout(0.2)(lstm1_2_lynorm)

    lstm1_3_out = Bidirectional(LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True))(dropout_2)
    lstm1_3_lynorm = LayerNormalization()(lstm1_3_out)

    # lstm1_4_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(dropout_3)
    # lstm1_4_lynorm = LayerNormalization()(lstm1_4_out)
    # dropout_4 = Dropout(0.2)(lstm1_4_lynorm)

    # Luong Attention
    attention_layer = Attention(units=256, score='luong')(lstm1_3_lynorm)
    # Keras Attention
    ## Solution 1.
    # pooled_output = GlobalAveragePooling1D()(lstm1_3_lynorm)
    # attention_layer = Attention()([pooled_output, pooled_output])
    ## Solution 2.
    # Assuming you want to calculate attention over the last LSTM output
    # query = lstm1_3_lynorm[:, -1, :]  # Take the last timestep (if appropriate)
    # query = tf.expand_dims(query, 1)  # Expand dims to fit attention requirements
    # # Now apply attention
    # attention_layer = Attention()([query, lstm1_3_lynorm])
    # attention_layer = tf.squeeze(attention_layer, axis=1)

    dropout_3 = Dropout(0.2)(attention_layer)
    outputs = Dense(1)(dropout_3)

    # intialize & compile
    model = Model(inputs=ts_input, outputs=outputs)

    # Set Learning rate decay for AdamOpt
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=30000,
        decay_rate=0.80,
        staircase=True
    )
    # CosineDecay with warmup
    # lr_schedule = (
    #     tf.keras.optimizers.schedules.CosineDecayRestarts(
    #         initial_learning_rate=0.1,
    #         first_decay_steps=15000,
    #     )
    # )

    Adam_Opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # # Compile the model
    # model.compile(optimizer=Adam_Opt, loss=q_loss.call)
    # Huber Loss
    model.compile(optimizer=Adam_Opt, loss=tf.keras.losses.Huber())
    # model.compile(optimizer=Adam_Opt, loss='mae')

    print(f'Model Output shape: {y.shape}')

    print(f'Model Summary: {model.summary()}')

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

    # Set up Tensorboard
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f'Tensorboard log path: {logdir}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    class ExportAttentionWeights(Callback):
        def on_epoch_end(self, epoch, logs=None):
            attention_map = get_activations(model, val_X[:128])['attention_weight']
            print(attention_map.shape)

    history = model.fit(train_X, train_y, epochs=n_epoch,
                        batch_size=n_batch,
                        verbose=1, shuffle=True,
                        validation_split=val,
                        # validation_data=([val_X1, val_X2], val_y),
                        callbacks=[tensorboard_callback, es,
                                   # ExportAttentionWeights()
                                   ])

    return model


def make_predictions(model, n_batch, inputs):
    predictions = model.predict(inputs, batch_size=n_batch)

    return predictions


def evaluate_forecasts(truth, pred):
    r_square = r2_score(truth, pred)
    mae = mean_absolute_error(truth, pred)
    mape = np.mean(np.abs((truth - pred) / truth)) * 100
    rmse = np.sqrt(mean_squared_error(truth, pred))
    mbe = np.mean(pred - truth)

    print(f"R2: {r_square}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")
    print(f"MBE: {mbe}")
    print('==============================')

    # Return dict for logging
    return {
        "R2": r_square,
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "MBE": mbe
    }


# Custom function to split data based on lat/lon
def create_spatial_folds(lat, lon, n_splits=10, random_state=42):
    """
    Create random folds based on unique lat/lon combinations.

    Args:
        lat (np.ndarray): Latitude array.
        lon (np.ndarray): Longitude array.
        n_splits (int): Number of folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Fold indices for each data point.
    """
    np.random.seed(random_state)

    # Combine lat/lon into a unique identifier for each location
    unique_locs = np.unique(np.stack([lat, lon], axis=1), axis=0)

    # Shuffle and assign each unique location to a fold
    np.random.shuffle(unique_locs)
    fold_assignments = np.arange(len(unique_locs)) % n_splits

    # Map folds back to the original dataset
    fold_mapping = {
        tuple(loc): fold for loc, fold in zip(unique_locs, fold_assignments)
    }

    # Assign folds to each data point
    folds = np.array([fold_mapping[tuple(loc)] for loc in np.stack([lat, lon], axis=1)])
    return folds


if __name__ == "__main__":
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(os.getenv('TF_GPU_ALLOCATOR'))
    # Set GPU Memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    """
    # Load Training Data
    X_path = sorted(glob.glob("../data/input_TS/TS_X_*.npy"))
    y_path = sorted(glob.glob("../data/input_TS/TS_y_*.npy"))

    print(f"Num. of Days of Training Set: {len(X_path)}")

    X = np.concatenate([np.load(daily_X_path) for daily_X_path in X_path])
    y = np.concatenate([np.load(daily_y_path) for daily_y_path in y_path])

    # X = np.load('../data/input_TS_merge/TS_X_all.npy')
    # y = np.load('../data/input_TS_merge/TS_y_all.npy')
    # # Adjust for inputs
    # dist_mask = np.array(X[:, :, 15] <= 500 * 1000, dtype=int)
    # dist_mask[dist_mask == 0] = -1
    # X[:, :, 15] = dist_mask
    # X[:, :, 16] = X[:, :, 16] * dist_mask
    # X[X[:, :, 16] == 0] = -1
    mask = list(range(0, 23))
    mask.remove(15)
    mask.remove(16)
    X = X[:, :, mask]

    # X = np.load('../data/input_TS_merge/TS_X_filter.npy')
    # y = np.load('../data/input_TS_merge/TS_y_filter.npy').reshape(-1,1)

    # Only keep non-negative records
    X = X[(y > 0).squeeze(), :, :]
    y = y[(y > 0).squeeze(), :]
    # Keep records without NaNs
    nonan_idx = np.argwhere(~np.isnan(X).any(axis=(1, 2))).squeeze()
    X = X[nonan_idx, :, :]
    y = y[nonan_idx, :]
    """

    # Prepare a list to hold logging info for each fold
    logs = []

    # Drop knnidw_distance and knnidw_pm25
    # mask = list(range(0, 23))
    mask = list(range(0, 23+4))
    mask.remove(15)
    mask.remove(16)
    # Drop lat/lon
    mask.remove(5)
    mask.remove(6)

    # Load CV Set
    year = 2021
    X = np.load(f'../data/X_{year}.npy')
    fold_idx = np.load(f'../data/X_{year}_checkerboard_fold2_deg16_idx.npy')
    y = np.load(f'../data/y_{year}.npy').reshape(-1, 1)

    # Apply Sinusoidal Encoding
    lat = np.deg2rad(X[:, :, 5])  # convert latitude to radians
    lon = np.deg2rad(X[:, :, 6])  # convert longitude to radians
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    X = np.concatenate([
        X,
        sin_lat[..., None],
        cos_lat[..., None],
        sin_lon[..., None],
        cos_lon[..., None]
    ], axis=2)

    # Only keep non-negative records
    X = X[(y > 0).squeeze(), :, :]
    fold_idx = fold_idx[(y > 0).squeeze()]
    y = y[(y > 0).squeeze(), :]


    train_size = X.shape[0]
    time_lag = X.shape[1]
    n_features = X.shape[2]

    print(f"Total Training Samples: {X.shape}, {y.shape}")

    # Normalize Inputs
    X, y, X_scaler, y_scaler = normalize_data(X, y)
    # Replace NaN with -1
    X = np.nan_to_num(X, nan=0)
    # Reshape X to LSTM format
    X = X.reshape(-1, time_lag, n_features)

    # Generate spatial folds
    lat = X[:, -1, 5].ravel()  # Flatten latitudes
    lon = X[:, -1, 6].ravel()  # Flatten longitudes
    X = X[:, :, mask]

    n_folds = sorted(np.unique(fold_idx))

    # Iterate over the folds
    for fold_num in n_folds:#range(10):
        # Get Spatial CV Index
        train_idx = np.where(fold_idx != fold_num)[0]
        test_idx = np.where(fold_idx == fold_num)[0]
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]

        # Calculate validation set ratio
        test_ratio = len(test_idx) / len(X)

        print(
            f"Fold Num: {fold_num} Train Size: {train_X.shape} | Test Size: {test_X.shape} | Test Ratio: {test_ratio:.2%}")

        if fold_num in []:
            fold_num += 1
            continue
        else:
            # Configure GPU settings
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)

            # Train the model
            model = fit_lstm_model(
                train_X=train_X, train_y=train_y,
                n_batch=2 ** 8, n_epoch=100,
                n_neurons=2 ** 8, val=0.05
            )
            print("Model Training Completed!")

            # Save the model
            model.save(f"../model/LSTM_Model/LSTM_{fold_num}.h5")
            print('Model Saved!')
            K.clear_session()

            # # Evaluate validation set
            # print("============= Validation =================")
            # pred_val = make_predictions(model=model, n_batch=2 ** 8, inputs=val_X)
            # inv_pred_val = y_scaler.inverse_transform(pred_val)
            # inv_y_val = y_scaler.inverse_transform(val_y)
            # val_metrics = evaluate_forecasts(truth=inv_y_val, pred=inv_pred_val)

            # Evaluate test set
            print("============= Test =================")
            pred_test = make_predictions(model=model, n_batch=2 ** 8, inputs=test_X)
            inv_pred_test = y_scaler.inverse_transform(pred_test)
            inv_y_test = y_scaler.inverse_transform(test_y)
            test_metrics = evaluate_forecasts(truth=inv_y_test, pred=inv_pred_test)

            # Save test predictions
            test_pred = pd.DataFrame({'truth': inv_y_test.ravel(),
                                      'bilstm': inv_pred_test.ravel()})
            test_pred.to_csv(f"../results/pred_fold_{fold_num}.csv", index=False)

            print("Training Finished!")

            # Log results
            logs.append({
                "fold": fold_num,
                "test_ratio": test_ratio,
                # "val_R2": val_metrics["R2"],
                # "val_MBE": val_metrics["MBE"],
                # "val_RMSE": val_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
                "test_MBE": test_metrics["MBE"],
                "test_RMSE": test_metrics["RMSE"]
            })

            fold_num += 1

    # Save logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv("../results/spatial_training_log.csv", index=False)
