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
    plt.show()


df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist()
train_labels = np.array(to_number(df.v1))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
tokenized_texts = tokenizer.texts_to_sequences(train_texts)

MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_texts)
tokenized_texts = pad_sequences(tokenized_texts , maxlen=MAX_LENGTH , padding="post")

model = Sequential()

model.add(Input(shape=(MAX_LENGTH ,)))
model.add(Embedding(vocab_size, 2, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd")
model.fit(tokenized_texts, np.arange(len(train_texts)), epochs=10, verbose=1)

# print(tokenizer.word_index)

# "xxx" "sex" "gay" "digital" "black" "high" "lady" "porn" "erotic","ramen","chick","boobs","revealing","bitching"])
words = [380, 890, 1297, 1305, 1306, 1378, 1570, 2296, 8743, 8790, 8817, 8818, 8819, 8849, 8920]

plot_vectors(model=model, tokenizer=tokenizer,indices=words)
