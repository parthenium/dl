# ----------------------------------------------------
# Time Series Forecasting using LSTM 
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Create synthetic time series data (sine wave)
time = np.arange(0, 400, 0.1)
series = np.sin(0.1 * time)

# Plot the series
plt.figure(figsize=(10, 4))
plt.plot(time, series)
plt.title("Original Time Series")
plt.show()

# 2. Prepare data using sliding window
def create_dataset(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 20
X, y = create_dataset(series, window_size)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 3. Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 4. Build LSTM model
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=False, input_shape=(window_size, 1)),
    Dense(1)
])

# 5. Compile model
model.compile(optimizer='adam', loss='mse')

# Summary
model.summary()

# 6. Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)

# 7. Evaluate model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Plot actual vs predicted
plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()
