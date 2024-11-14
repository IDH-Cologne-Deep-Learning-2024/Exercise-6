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
