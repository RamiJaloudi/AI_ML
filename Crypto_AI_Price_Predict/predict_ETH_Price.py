import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to fetch Ethereum prices from CoinGecko
def fetch_ethereum_prices():
    url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=max&interval=daily'
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1,1))
    return scaled_data, scaler

# Create dataset for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Predict future prices
def predict_future_prices(model, data, scaler, days=1825, look_back=1):
    future_prices = []
    current_batch = data[-look_back:]
    for i in range(days // 30):  # Predict monthly for 5 years
        current_pred = model.predict(current_batch.reshape(1, look_back, 1))
        future_prices.append(scaler.inverse_transform(current_pred)[0][0])
        current_batch = np.append(current_batch[1:], current_pred)
    return future_prices

df = fetch_ethereum_prices()
scaled_data, scaler = preprocess_data(df)
X, Y = create_dataset(scaled_data, look_back=1)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Build and train LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=20, batch_size=32, verbose=1)

# Predict future prices
future_prices = predict_future_prices(model, scaled_data, scaler)

# Print predicted prices
for i, price in enumerate(future_prices, 1):
    print(f"Month {i}: ${price:.2f}")

# Save predicted prices to CSV
pd.DataFrame(future_prices, columns=['Predicted Price']).to_csv('predicted_ethereum_prices.csv', index=False)

# Plot histogram of the predicted prices
plt.hist(future_prices, bins=30)
plt.title('Histogram of Projected Ethereum Prices Over the Next 5 Years')
plt.xlabel('Price in USD')
plt.ylabel('Frequency')
plt.show()