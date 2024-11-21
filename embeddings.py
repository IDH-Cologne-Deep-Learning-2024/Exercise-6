import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_number(labels):
    number_labels = []
    for label in labels:
        if label == "ham":
            number_labels.append(0)
        elif label == "spam":
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
        plt.quiver(
            np.array([0, 0]),
            np.array([0, 0]),
            vector[0],
            vector[1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        plt.text(vector[0], vector[1], label)
    plt.show()


df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist()
train_labels = np.array(to_number(df.v1))

# preprocessing the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
tok_texts = tokenizer.texts_to_sequences(train_texts)
vocab_length = len(tokenizer.word_index) + 1
max_length = max(len(seq) for seq in tok_texts)
pad_texts = pad_sequences(tok_texts, maxlen=max_length, padding="post")


# feed-forward neural network
FFNN = Sequential(
    [
        Embedding(input_dim=vocab_length, output_dim=2),
        Flatten(),
        Dense(15, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

FFNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
FFNN.fit(pad_texts, train_labels, batch_size=32, epochs=25, verbose=1)

# plotting vectors with selected words
# print(tokenizer.word_index)
# sel_words are "pls", "sorry", "tomorrow", "ready", "hope" "alright", "you","any"
sel_words = [104, 83, 151, 315, 126, 518, 3, 105]
plot_vectors(FFNN, tokenizer, sel_words)
