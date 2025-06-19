import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- Config ---
TICKER = 'MSFT'
CSV_PATH = 'msft_data.csv'
MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.save'
TIME_STEP = 60
EPOCHS = 3
BATCH_SIZE = 32

# --- Step 1: Fetch new data ---
def fetch_latest_close(ticker):
    data = yf.download(ticker, period='2d')  # get last 2 days to be safe
    return data['Close']

# --- Step 2: Update dataset ---
def update_dataset(csv_path, new_data):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        df = pd.DataFrame(columns=['Close'])

    for date, price in new_data.items():
        if date not in df.index:
            df.loc[date] = price

    df.sort_index(inplace=True)
    df.to_csv(csv_path)
    return df

# --- Step 3: Prepare sequences ---
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

# --- Step 4: Build model if none exists ---
def build_model(time_step):
    model = Sequential()
    model.add(LSTM(32, input_shape=(time_step,1)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Step 5: Retrain ---
def retrain_model(csv_path, model_path, scaler_path, time_step, epochs, batch_size):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    close_prices = df['Close'].values.reshape(-1,1)

    # Load or create scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(close_prices)
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        joblib.dump(scaler, scaler_path)

    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Load or build model
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model(time_step)

    # Train the model
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False, callbacks=[early_stop])

    model.save(model_path)
    print("Model retrained and saved.")
    return model, scaler

# --- Step 6: Predict future ---
def predict_future(model, scaler, data, time_step=TIME_STEP, future_steps=10):
    last_data = data[-time_step:]
    last_data = last_data.reshape(1, time_step, 1)
    future_preds = []

    for _ in range(future_steps):
        next_pred = model.predict(last_data)
        future_preds.append(next_pred[0,0])
        last_data = np.append(last_data[:, 1:, :], [[[next_pred[0,0]]]], axis=1)

    future_preds = np.array(future_preds).reshape(-1,1)
    future_preds = scaler.inverse_transform(future_preds)
    return future_preds

# --- Main execution ---

if __name__ == "__main__":
    # Fetch new close price data
    latest_data = fetch_latest_close(TICKER)
    print(f"Fetched latest data:\n{latest_data.tail()}")

    # Update CSV dataset
    df = update_dataset(CSV_PATH, latest_data)
    print(f"Dataset updated. Total records: {len(df)}")

    # Retrain model
    model, scaler = retrain_model(CSV_PATH, MODEL_PATH, SCALER_PATH, TIME_STEP, EPOCHS, BATCH_SIZE)

    # Prepare scaled data for prediction
    scaled_close = scaler.transform(df['Close'].values.reshape(-1,1))

    # Predict next 10 days future prices
    future_preds = predict_future(model, scaler, scaled_close, TIME_STEP, future_steps=10)
    print(f"Future 10-day predictions:\n{future_preds.flatten()}")
