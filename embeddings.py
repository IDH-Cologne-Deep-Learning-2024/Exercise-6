import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'ham':
            number_labels.append(0)
        elif label == 'spam':
            number_labels.append(1)
    return number_labels


def get_vocab_indices(tokenizer):
    for i, word in enumerate(["[PAD]"] + list(tokenizer.word_index.keys())):
        print(f"{word}: {i}")


def plot_vectors(model, tokenizer, indices=[0, 1, 2, 3, 4]):
    embeddings = model.layers[0].get_weights()
    embeddings = map(embeddings[0].tolist().__getitem__, indices)
    vocab = ["[PAD]"] + list(tokenizer.word_index.keys())
    vocab = map(vocab.__getitem__, indices)
    for vector, label in zip(embeddings, vocab):
        plt.quiver(np.array([0, 0]), np.array([0, 0]), vector[0], vector[1], angles='xy', scale_units='xy', scale=1)
        plt.text(vector[0], vector[1], label)
    plt.show()

# Loading and processing the data
df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist()
train_labels = np.array(to_number(df.v1))

# Tokenizing and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
vocab_size = len(tokenizer.word_index) + 1

# Building the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=2, input_length=max_len, name="embedding"),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compiling
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training
history = model.fit(padded_sequences, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Visualizing 
selected_words = ["free", "win", "call", "urgent", "hello", "love"]
indices = [tokenizer.word_index[word] for word in selected_words if word in tokenizer.word_index]
plot_vectors(model, tokenizer, indices)
