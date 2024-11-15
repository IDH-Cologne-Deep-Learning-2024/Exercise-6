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
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.show()


df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist()
train_labels = np.array(to_number(df.v1))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(train_texts)
max_length = max(len(sequence) for sequence in train_sequences)
tokenized_texts = pad_sequences(train_sequences, maxlen=max_length, padding='post')

FFNN = Sequential()
FFNN.add(Input(shape=(max_length,)))
FFNN.add(Embedding(vocab_size, 2, input_length=max_length))
FFNN.add(Flatten())
FFNN.add(Dense(4, activation='sigmoid'))
FFNN.add(Dense(1, activation='sigmoid'))
FFNN.compile(loss='binary_crossentropy', optimizer='sgd')

FFNN.fit(x=tokenized_texts, y=train_labels, epochs=10)
FFNN.summary()

# Standard Input
plot_vectors(FFNN, tokenizer)
# Plot first 200 Words
plot_vectors(FFNN, tokenizer, list(tokenizer.word_index.values())[:200])
# Plot all Words -> takes for me about 2 Minutes to load
# plot_vectors(FFNN, tokenizer, tokenizer.word_index.values())