import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

# ----------------------------------------------------------
# 1. Load MNIST dataset
# ----------------------------------------------------------
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize images (0–1 range)
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Flatten images (28×28 → 784)
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# ----------------------------------------------------------
# 2. Build a Simple Autoencoder
# ----------------------------------------------------------
encoding_dim = 32  # compressed representation dimension

autoencoder = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(encoding_dim, activation='relu'),   # bottleneck (compressed)
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')          # output same size as input
])

autoencoder.compile(optimizer='adam', loss='mse')

# ----------------------------------------------------------
# 3. Train the Autoencoder
# ----------------------------------------------------------
history = autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=256,
    validation_data=(x_test, x_test)
)

# ----------------------------------------------------------
# 4. Plot Training vs Validation Loss
# ----------------------------------------------------------
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# ----------------------------------------------------------
# 5. Reconstruct Test Images
# ----------------------------------------------------------
reconstructed = autoencoder.predict(x_test)

# ----------------------------------------------------------
# 6. Display Original vs Reconstructed Images
# ----------------------------------------------------------
n = 10
plt.figure(figsize=(10, 4))

for i in range(n):
    # Original
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].reshape(28,28), cmap="gray")
    plt.axis("off")

plt.suptitle("Original (Top) vs Reconstructed Images (Bottom)")
plt.show()
