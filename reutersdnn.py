import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load Reuters dataset
# -----------------------------
# num_words=10000 → keep the top 10k most frequent words
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)


print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# -----------------------------
# 2. Pad sequences to equal length
# -----------------------------
max_length = 200
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)


# -----------------------------
# 3. Number of output classes
# -----------------------------
num_classes = max(y_train) + 1   # Reuters has 46 classes


# -----------------------------
# 4. Build Deep Neural Network
# -----------------------------
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    Flatten(),


    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')    # multi-class classification
])


# -----------------------------
# 5. Compile the model
# -----------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


# -----------------------------
# 6. Train the network
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


# -----------------------------
# 7. Evaluate on test data
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", accuracy)


# -----------------------------
# 8. Plot accuracy & loss graphs
# -----------------------------


# Accuracy
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 9. Predict on a news sample
# -----------------------------
sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)
predicted_class = pred.argmax()


print("\nPredicted Topic:", predicted_class)
print("True Topic:", y_test[0])
