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


def fit_lstm_model(train_X, train_X_emb, train_y,
                   n_batch, n_epoch, n_neurons, val=0.05):
    # 1) Define the time-series input
    ts_input = Input(shape=(train_X.shape[1], train_X.shape[2]), name='ts_input')
    mask_layer = tf.keras.layers.Masking(mask_value=-1)(ts_input)

    # LSTM stack
    lstm1_1_out = Bidirectional(LSTM(int(n_neurons), activation="tanh", return_sequences=True))(mask_layer)
    lstm1_1_lynorm = LayerNormalization()(lstm1_1_out)
    dropout_1 = Dropout(0.2)(lstm1_1_lynorm)

    lstm1_2_out = Bidirectional(LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True))(dropout_1)
    lstm1_2_lynorm = LayerNormalization()(lstm1_2_out)
    dropout_2 = Dropout(0.2)(lstm1_2_lynorm)

    lstm1_3_out = Bidirectional(LSTM(int(n_neurons / 2), activation="tanh", return_sequences=True))(dropout_2)
    lstm1_3_lynorm = LayerNormalization()(lstm1_3_out)

    # 2) Attention layer (256-d output)
    attention_layer = Attention(units=256, score='luong')(lstm1_3_lynorm)
    # shape: (batch_size, 256)

    dropout_3 = Dropout(0.2)(attention_layer)

    # 3) Location embedding input & projection
    loc_input = Input(shape=(512,), name='location_input')
    loc_proj = Dense(256, activation='relu', name='loc_proj')(loc_input)
    # loc_proj = LayerNormalization()(loc_proj)
    # shape: (batch_size, 256)

    # 4) Dot product
    dot_fusion = tf.keras.layers.Multiply(name='fusion_hadamard')([dropout_3, loc_proj])
    # dot_fusion = tf.keras.layers.Concatenate(name='fusion_concat')([dropout_3, loc_proj])
    # shape: (batch_size, 256)

    # 5) Dense(256) -> final regression
    fusion_fc = Dense(256, activation='relu', name='fusion_fc')(dot_fusion)
    outputs = Dense(1, name='final_output')(fusion_fc)

    # 6) Define model with 2 inputs
    model = Model(inputs=[ts_input, loc_input], outputs=outputs, name='LSTM_with_loc_fusion')

    # 7) Compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=30000,
        decay_rate=0.80,
        staircase=True
    )
    Adam_Opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=Adam_Opt, loss=tf.keras.losses.Huber())
    print(model.summary())

    # 8) Fit the model => pass [train_X, train_X_emb] as input
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1)

    # Set up Tensorboard
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f'Tensorboard log path: {logdir}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit([train_X, train_X_emb],
                        train_y,
                        epochs=n_epoch,
                        batch_size=n_batch,
                        verbose=1,
                        shuffle=True,
                        validation_split=val,
                        callbacks=[tensorboard_callback, es])

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
    mask = list(range(0, 23))
    mask.remove(15)
    mask.remove(16)
    # Drop lat/lon
    mask.remove(5)
    mask.remove(6)

    # Load CV Set
    year = 2021
    CV_X = np.load(f'../data/X_{year}_CV_spatial.npy')
    CV_X_emb = np.load(f'../data/X_{year}_CV_spatial_geoclip.npy')[:, 2:]
    CV_y = np.load(f'../data/y_{year}_CV_spatial.npy').reshape(-1, 1)

    # Only keep non-negative records
    CV_X = CV_X[(CV_y > 0).squeeze(), :, :]
    CV_X_emb = CV_X_emb[(CV_y > 0).squeeze(), :]
    CV_y = CV_y[(CV_y > 0).squeeze(), :]

    # Load Test Set
    test_X = np.load('../data/X_2021_test_spatial.npy')
    test_X_emb = np.load('../data/X_2021_test_spatial_geoclip.npy')[:, 2:]
    test_y = np.load('../data/y_2021_test_spatial.npy').reshape(-1, 1)

    # Only keep non-negative records
    test_X = test_X[(test_y > 0).squeeze(), :, :]
    test_X_emb = test_X_emb[(test_y > 0).squeeze(), :]
    test_y = test_y[(test_y > 0).squeeze(), :]

    X = np.concatenate([CV_X, test_X])
    y = np.concatenate([CV_y, test_y])

    train_size = CV_X.shape[0]
    emb_size = CV_X_emb.shape[0]
    time_lag = X.shape[1]
    n_features = X.shape[2]

    print(f"Total Training Samples: {CV_X.shape}, {CV_X_emb.shape}, {CV_y.shape}")

    # Normalize Inputs
    X, y, X_scaler, y_scaler = normalize_data(X, y)
    # Replace NaN with -1
    X = np.nan_to_num(X, nan=0)
    # Reshape X to LSTM format
    X = X.reshape(-1, time_lag, n_features)

    test_X = X[train_size:]
    test_y = y[train_size:]
    X = X[:train_size]
    y = y[:train_size]

    # Generate spatial folds
    lat = X[:, -1, 5].ravel()  # Flatten latitudes
    lon = X[:, -1, 6].ravel()  # Flatten longitudes
    X = X[:, :, mask]
    test_X = test_X[:, :, mask]
    spatial_folds = create_spatial_folds(lat, lon, n_splits=10, random_state=42)

    fold_num = 0
    # Iterate over the folds
    for fold_num in range(10):
        # Get Spatial CV Index
        val_idx = np.where(spatial_folds == fold_num)[0]
        train_idx = np.where(spatial_folds != fold_num)[0]
        train_X = X[train_idx]
        train_X_emb = CV_X_emb[train_idx]
        train_y = y[train_idx]
        test_X = test_X
        test_y = test_y

        # Use spatial folds to create validation set
        val_X = X[val_idx]
        val_X_emb = CV_X_emb[val_idx]
        val_y = y[val_idx]

        # Calculate validation set ratio
        val_ratio = len(val_idx) / len(CV_X)

        print(
            f"Fold Num: {fold_num} Train Size: {train_X.shape} | Val Size: {val_X.shape} | Val Ratio: {val_ratio:.2%}")

        if fold_num in []:
            fold_num += 1
            continue
        else:

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

            model = fit_lstm_model(
                train_X=train_X,  # shape: (batch_size, 21, 19)
                train_X_emb=train_X_emb,  # shape: (batch_size, 512)
                train_y=train_y,
                n_batch=512,
                n_epoch=100,
                n_neurons=256,
                val=0.05
            )

            print("Model Training Completed!")
            # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
            model.save(f"../model/LSTM_Model/LSTM_{fold_num}.h5")
            print('Model Saved!')

            K.clear_session()

            # Evaluate Train
            # print("============= Train =================")
            # pred_train = make_predictions(model=model, n_batch=2 ** 8, inputs=train_X)
            #
            # print('Inverse forecast to unscaled numbers!')
            #
            # inv_pred_train = y_scaler.inverse_transform(pred_train)
            # inv_y = y_scaler.inverse_transform(train_y)
            #
            # evaluate_forecasts(truth=inv_y, pred=inv_pred_train)

            # Evaluate validation
            print("============= Validation =================")
            pred_val = make_predictions(model=model, n_batch=2 ** 9, inputs=[val_X, val_X_emb])

            print('Inverse forecast to unscaled numbers!')

            inv_pred_val = y_scaler.inverse_transform(pred_val)
            inv_y_val = y_scaler.inverse_transform(val_y)

            val_metrics = evaluate_forecasts(truth=inv_y_val, pred=inv_pred_val)

            # Evaluate test
            print("============= Test =================")
            pred_test = make_predictions(model=model, n_batch=2 ** 9, inputs=[test_X, test_X_emb])

            print('Inverse forecast to unscaled numbers!')

            inv_pred_test = y_scaler.inverse_transform(pred_test)
            inv_y_test = y_scaler.inverse_transform(test_y)

            test_metrics = evaluate_forecasts(truth=inv_y_test, pred=inv_pred_test)

            test_pred = pd.DataFrame({'truth': inv_y_test.ravel(),
                                      'bilstm': inv_pred_test.ravel()})
            test_pred.to_csv(f"../results/pred_fold_{fold_num}.csv", index=False)

            print("Training Finished!")

            # Append this fold's results to our logs list
            logs.append({
                "fold": fold_num,
                "val_R2": val_metrics["R2"],
                "val_MBE": val_metrics["MBE"],
                "val_RMSE": val_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
                "test_MBE": test_metrics["MBE"],
                "test_RMSE": test_metrics["RMSE"]
            })

            fold_num += 1

    # Once all folds are done, convert to DataFrame and save
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv("../results/training_log.csv", index=False)
