# --------------------------------------------------
# 📌 1. IMPORT LIBRARIES
# --------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# --------------------------------------------------
# 📌 2. LOAD IMDB DATASET (Already tokenized)
# --------------------------------------------------
# num_words=10000 → keep the 10k most frequent words
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)


# --------------------------------------------------
# 📌 3. PAD SEQUENCES TO SAME LENGTH
# --------------------------------------------------
# Ensure all reviews = length 200
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=200)


# --------------------------------------------------
# 📌 4. BUILD FEEDFORWARD NEURAL NETWORK
# --------------------------------------------------
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=32, input_length=200),
    layers.GlobalAveragePooling1D(),   # Converts sequence → fixed-size vector
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: probability positive
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


print(model.summary())


# --------------------------------------------------
# 📌 5. TRAIN THE MODEL
# --------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# 📌 6. EVALUATE THE MODEL
# --------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")


# --------------------------------------------------
# 📌 7. FUNCTION TO PREDICT REVIEW SENTIMENT
# --------------------------------------------------


# Load IMDB word index (mapping words → numbers)
word_index = tf.keras.datasets.imdb.get_word_index()


# Reverse mapping (numbers → words)
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"




def encode_review(text):
    """Convert a string review to numerical sequence using IMDB word index."""
    words = text.lower().split()
    encoded = [1]  # <START>
    
    for word in words:
        if word in word_index:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # <UNK>
    
    return tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=200)




def predict_sentiment(review):
    encoded = encode_review(review)
    prediction = model.predict(encoded)[0][0]
    
    if prediction >= 0.5:
        print(f"\n🙂 SENTIMENT: POSITIVE ({prediction:.2f})")
    else:
        print(f"\n☹ SENTIMENT: NEGATIVE ({1-prediction:.2f})")




# --------------------------------------------------
# 📌 8. EXAMPLE PREDICTION
# --------------------------------------------------


sample_review = "The movie was absolutely wonderful, amazing acting and great story!"
predict_sentiment(sample_review)
  
