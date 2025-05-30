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


def permutation_importance_lstm(
        model,
        X_test,
        y_test,
        n_batch,
        y_scaler,
        evaluate_func,
        n_repeats=3,
        random_state=42,
        metric_key="R2"
):
    """
    Compute Permutation Feature Importance for a Keras LSTM model on a 3D dataset.
    Shuffles each feature across (samples, time_lag).

    Parameters
    ----------
    model : tf.keras.Model
        A trained Keras model.
    X_test : np.ndarray
        Test features, shape = (n_samples, time_lag, n_features).
    y_test : np.ndarray
        Test labels, shape = (n_samples, 1).
    n_batch : int
        Batch size for predictions.
    y_scaler : sklearn Scaler
        Scaler to invert-transform predictions and labels.
    evaluate_func : function
        e.g. `evaluate_forecasts(truth, pred)` returning a dict with "R2", "RMSE", etc.
    n_repeats : int
        Number of shuffles per feature to reduce variance (the code will compute mean & std).
    random_state : int
        Random seed for reproducibility.
    metric_key : str
        Key in the evaluate_func output to measure drop (e.g. "R2", "RMSE").

    Returns
    -------
    importances_mean : np.ndarray
        Shape (n_features,). The average drop in the chosen metric (baseline - permuted).
    importances_std : np.ndarray
        Shape (n_features,). The standard deviation of the drop across `n_repeats`.
        This is only relevant if n_repeats > 1.
    """

    rng = np.random.default_rng(random_state)

    # 1. Baseline performance
    baseline_pred = make_predictions(model, n_batch, X_test)
    inv_baseline_pred = y_scaler.inverse_transform(baseline_pred)
    inv_y_true = y_scaler.inverse_transform(y_test)

    baseline_metrics = evaluate_func(truth=inv_y_true, pred=inv_baseline_pred)
    baseline_score = baseline_metrics[metric_key]

    n_features = X_test.shape[2]
    importances_mean = np.zeros(n_features, dtype=np.float64)
    importances_std = np.zeros(n_features, dtype=np.float64)

    # For R2 we assume: drop = baseline_score - permuted_score
    # If you pick RMSE or MAE (lower is better), invert that logic as needed.

    for f in range(n_features):
        saved_feature = X_test[:, :, f].copy()
        drop_scores = []

        for _ in range(n_repeats):
            # Shuffle this feature
            flat_col = X_test[:, :, f].reshape(-1)
            rng.shuffle(flat_col)
            X_test[:, :, f] = flat_col.reshape(X_test[:, :, f].shape)

            # Evaluate
            perm_pred = make_predictions(model, n_batch, X_test)
            inv_perm_pred = y_scaler.inverse_transform(perm_pred)
            perm_metrics = evaluate_func(truth=inv_y_true, pred=inv_perm_pred)
            perm_score = perm_metrics[metric_key]

            drop_scores.append(baseline_score - perm_score)

            # Restore
            X_test[:, :, f] = saved_feature

        importances_mean[f] = np.mean(drop_scores)
        importances_std[f] = np.std(drop_scores, ddof=1)  # sample std

    return importances_mean, importances_std





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
    CV_X = np.load(f'../data/X_{year}_CV.npy')
    CV_y = np.load(f'../data/y_{year}_CV.npy').reshape(-1, 1)

    # Only keep non-negative records
    CV_X = CV_X[:, :, mask]
    CV_X = CV_X[(CV_y > 0).squeeze(), :, :]
    CV_y = CV_y[(CV_y > 0).squeeze(), :]

    # Load Test Set
    test_X = np.load('../data/X_2021_test.npy')
    test_y = np.load('../data/y_2021_test.npy').reshape(-1, 1)

    # Only keep non-negative records
    test_X = test_X[:, :, mask]
    test_X = test_X[(test_y > 0).squeeze(), :, :]
    test_y = test_y[(test_y > 0).squeeze(), :]

    X = np.concatenate([CV_X, test_X])
    y = np.concatenate([CV_y, test_y])

    train_size = CV_X.shape[0]
    time_lag = X.shape[1]
    n_features = X.shape[2]

    print(f"Total Training Samples: {CV_X.shape}, {CV_y.shape}")

    # Normalize Inputs
    X, y, X_scaler, y_scaler = normalize_data(X, y)
    # Replace NaN with -1
    X = np.nan_to_num(X, nan=0)
    # Reshape X to LSTM format
    X = X.reshape(-1, time_lag, n_features)

    test_X = X[train_size:]
    test_y = y[train_size:]
    train_X = X[:train_size]
    train_y = y[:train_size]

    ##################
    # Train the model
    print("===== Training Model =====")
    model = fit_lstm_model(
        train_X=train_X,
        train_y=train_y,
        n_batch=2 ** 8,
        n_epoch=100,
        n_neurons=2 ** 8,
        val=0.05
    )

    print("Model Training Completed!")
    model.save("../model/LSTM_Model/LSTM_SingleFold_wo_latlon.h5")
    print("Model Saved!")

    # Evaluate on Test
    print("===== Evaluate on Test Set =====")
    pred_test = make_predictions(model=model, n_batch=2 ** 8, inputs=test_X)
    inv_pred_test = y_scaler.inverse_transform(pred_test)
    inv_y_test = y_scaler.inverse_transform(test_y)

    test_metrics = evaluate_forecasts(truth=inv_y_test, pred=inv_pred_test)
    print("Test Metrics:", test_metrics)

    # Optionally, save predictions
    test_pred_df = pd.DataFrame({
        'truth': inv_y_test.ravel(),
        'bilstm': inv_pred_test.ravel()
    })
    test_pred_df.to_csv("../results/test_predictions_singlefold_wo_latlon.csv", index=False)

    ##################
    # Permutation Importance
    # Increase n_repeats if you want more robust estimates.
    print("===== Permutation Feature Importance =====")
    importances_mean, importances_std = permutation_importance_lstm(
        model=model,
        X_test=test_X,
        y_test=test_y,
        n_batch=2 ** 8,
        y_scaler=y_scaler,
        evaluate_func=evaluate_forecasts,
        n_repeats=5,  # e.g., shuffle 5 times per feature
        random_state=42,
        metric_key="RMSE"  # or "RMSE", "MAE", etc. If you choose "RMSE"/"MAE", invert the drop logic
    )

    # Build a DataFrame for easy reporting
    feat_names = [f"Feature_{i}" for i in range(n_features)]
    importance_df = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": importances_mean,
        "importance_std": importances_std
    })

    # Sort by importance mean (descending order)
    importance_df.sort_values("importance_mean", ascending=False, inplace=True)
    importance_df.reset_index(drop=True, inplace=True)

    print("\n=== Permutation Feature Importance Results ===")
    print(importance_df)

    # Save to CSV
    importance_df.to_csv("../results/feature_importance_wo_latlon.csv", index=False)
    print("Feature importance saved to ../results/feature_importance.csv")

    print("Done.")
