import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


import tensorflow as tf
from tensorflow.keras import layers




# -----------------------------------------
# 📌 LOAD BOSTON HOUSING DATASET
# -----------------------------------------


boston = fetch_openml(name='boston', version=1, as_frame=True)


X = boston.data
y = boston.target.to_numpy().astype("float32")




# -----------------------------------------
# 📌 TRAIN-TEST SPLIT
# -----------------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# -----------------------------------------
# 📌 FEATURE SCALING
# -----------------------------------------


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# -----------------------------------------
# 📌 BUILD DEEP FEEDFORWARD NN
# -----------------------------------------


model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)




# -----------------------------------------
# 📌 TRAIN THE MODEL
# -----------------------------------------


history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)




# -----------------------------------------
# 📌 MODEL EVALUATION
# -----------------------------------------


loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Mean Absolute Error (MAE): {mae:.2f}")




# -----------------------------------------
# 📌 GRAPH 1 — LOSS CURVE (MSE)
# -----------------------------------------


plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Training vs Validation Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()




# -----------------------------------------
# 📌 GRAPH 2 — MAE CURVE
# -----------------------------------------


plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label="Training MAE")
plt.plot(history.history['val_mae'], label="Validation MAE")
plt.title("Training vs Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()




# -----------------------------------------
# 📌 PREDICTION FUNCTION
# -----------------------------------------


def predict_price(**kwargs):


    input_df = pd.DataFrame([kwargs])
    input_scaled = scaler.transform(input_df)


    price = model.predict(input_scaled)[0][0]
    print(f"\n🏠 Predicted House Price: ${price:.2f} (in 1000s USD)")
    return price




# Example prediction (modify values)
predict_price(
    CRIM=0.1, ZN=12, INDUS=7, CHAS=0, NOX=0.5,
    RM=6.3, AGE=60, DIS=3.5, RAD=1, TAX=300,
    PTRATIO=15, B=395, LSTAT=4
)
