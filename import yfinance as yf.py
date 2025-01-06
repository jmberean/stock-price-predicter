import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go

def set_random_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def download_data(ticker="SPY", period="1y"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def scale_data(df, features, target):
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    
    scaled_inputs = input_scaler.fit_transform(df[features])
    scaled_target = output_scaler.fit_transform(df[[target]])
    
    return scaled_inputs, scaled_target, input_scaler, output_scaler

def prepare_sequences(scaled_inputs, scaled_target, look_back, forecast_horizon):
    X, Y = [], []
    for i in range(len(scaled_inputs) - look_back - forecast_horizon):
        X.append(scaled_inputs[i:i + look_back])
        Y.append(scaled_target[i + look_back:i + look_back + forecast_horizon].flatten())
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def split_data(X, Y, split_ratio=0.8):
    split = int(len(X) * split_ratio)
    return X[:split], X[split:], Y[:split], Y[split:]

def build_model(input_shape, forecast_horizon):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(forecast_horizon)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=200, batch_size=64):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-5)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop, reduce_lr]
    )
    return history

def evaluate_model(Y_test, Y_pred, scaler):
    Y_test_inv = scaler.inverse_transform(Y_test)
    Y_pred_inv = scaler.inverse_transform(Y_pred)
    
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(Y_test_inv, Y_pred_inv)),
        "MAE": mean_absolute_error(Y_test_inv, Y_pred_inv),
        "MAPE": mean_absolute_percentage_error(Y_test_inv, Y_pred_inv),
        "R2": r2_score(Y_test_inv, Y_pred_inv)
    }
    return metrics, Y_test_inv, Y_pred_inv

def plot_results_with_forecast(dates, actual_prices, test_pred_inv, test_dates, future_pred_inv, future_dates):
    # Ensure data is formatted properly
    dates = pd.to_datetime(dates)
    test_dates = pd.to_datetime(test_dates)
    future_dates = pd.to_datetime(future_dates)
    
    # Flatten the arrays to 1D
    actual_prices = np.array(actual_prices).flatten()
    test_pred_inv = np.array(test_pred_inv).flatten()
    future_pred_inv = np.array(future_pred_inv).flatten()

    # Ensure numeric conversion
    actual_prices = pd.to_numeric(actual_prices)
    test_pred_inv = pd.to_numeric(test_pred_inv)
    future_pred_inv = pd.to_numeric(future_pred_inv)

    # Create the Plotly graph
    fig = go.Figure()

    # Add actual prices
    fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Actual Prices'))

    # Add test predictions
    fig.add_trace(go.Scatter(x=test_dates, y=test_pred_inv, mode='lines', name='Test Predictions'))

    # Add future forecast
    fig.add_trace(go.Scatter(x=future_dates, y=future_pred_inv, mode='lines', name='Future Forecast', line=dict(dash='dash')))

    # Customize layout
    fig.update_layout(
        title="Actual vs Predicted Close Prices (with Future Forecast)",
        xaxis_title="Date",
        yaxis_title="Close Price",
        legend_title="Legend",
        template="plotly_white"
    )

    fig.show()

def predict_future(model, scaler, scaled_inputs, look_back, forecast_horizon, last_date):
    last_seq = np.expand_dims(scaled_inputs[-look_back:], axis=0)
    future_pred_scaled = model.predict(last_seq)
    future_pred = scaler.inverse_transform(future_pred_scaled).flatten()
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
    results = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_pred})
    return results, future_dates

if __name__ == "__main__":
    # Set seeds for reproducibility
    set_random_seed()

    # Download and prepare data
    df = download_data()
    input_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled_inputs, scaled_close, input_scaler, output_scaler = scale_data(df, input_features, 'Close')

    look_back = 60
    forecast_horizon = 7
    X, Y = prepare_sequences(scaled_inputs, scaled_close, look_back, forecast_horizon)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Build and train the model
    model = build_model(input_shape=(look_back, len(input_features)), forecast_horizon=forecast_horizon)
    history = train_model(model, X_train, Y_train, X_test, Y_test)

    # Evaluate the model
    Y_pred_scaled = model.predict(X_test)
    metrics, Y_test_inv, Y_pred_inv = evaluate_model(Y_test, Y_pred_scaled, output_scaler)

    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Predict future prices
    future_results, future_dates = predict_future(model, output_scaler, scaled_inputs, look_back, forecast_horizon, df.index[-1])
    future_pred_inv = future_results['Predicted_Close'].values

    # Print future predictions
    print("\nFuture Predictions:")
    print(future_results)

    # Extract test dates for plotting
    test_dates = df.index[-len(Y_test):]

    # Plot results using actual close prices, test predictions, and future forecasts
    actual_prices = df['Close'].values
    plot_results_with_forecast(
        dates=df.index, 
        actual_prices=actual_prices, 
        test_pred_inv=Y_pred_inv, 
        test_dates=test_dates, 
        future_pred_inv=future_pred_inv, 
        future_dates=future_dates
    )


# Model Evaluation Metrics:
# RMSE: 11.171177384796898
# MAE: 9.199509620666504
# MAPE: 0.015416008420288563
# R2: -1.0738651752471924
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step

# Future Predictions:
#         Date  Predicted_Close
# 0 2025-01-06       589.299072
# 1 2025-01-07       599.985962
# 2 2025-01-08       599.384033
# 3 2025-01-09       609.643250
# 4 2025-01-10       605.563171
# 5 2025-01-13       595.013611
# 6 2025-01-14       607.814758