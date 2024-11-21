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


df = pd.read_csv("spam.csv", encoding="latin-1")
train_labels = np.array(to_number(df.v1))
train_texts = df.v2.tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_texts = tokenizer.texts_to_sequences(train_texts)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_texts)
tokenized_texts = pad_sequences(tokenized_texts, maxlen=MAX_LENGTH, padding="post")

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 2, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd")
model.fit(tokenized_texts, train_labels, epochs=50, verbose=1)
embeddings = model.layers[0].get_weights()

print(tokenized_texts)
print(embeddings)


def get_vocab_dict(tokenizer):
    vocab_dict = {}
    for i, word in enumerate(["[PAD]"] + list(tokenizer.word_index.keys())):
        vocab_dict[word] = i
    return vocab_dict


def plot_vectors(model, tokenizer, words=["you", "i"]):
    embeddings = model.layers[0].get_weights()
    vocab_dict = get_vocab_dict(tokenizer)
    indices = [vocab_dict[word] for word in words]
    embeddings = map(embeddings[0].tolist().__getitem__, indices)
    vocab = list(vocab_dict.keys())
    vocab = map(vocab.__getitem__, indices)
    for vector, label in zip(embeddings, vocab):
        plt.quiver(np.array([0, 0]), np.array([0, 0]), vector[0], vector[1], angles='xy', scale_units='xy', scale=1)
        plt.text(vector[0], vector[1], label)
    plt.show()


plot_vectors(model, tokenizer, words=["brown", "purple", "yellow", "london", "chinatown", "lancaster", "nottingham", "africa", "tamilnadu", "i", "you", "a", "the", "u", "and", "is", "brother", "sister", "boy", "girl"])
print(tokenizer.word_index)
