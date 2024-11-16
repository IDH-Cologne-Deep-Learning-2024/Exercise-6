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
    embeddings = list(map(embeddings[0].tolist().__getitem__, indices))
    vocab = ["[PAD]"] + list(tokenizer.word_index.keys())
    vocab = list(map(vocab.__getitem__, indices))
    for vector, label in zip(embeddings, vocab):
        plt.quiver(
            np.array([0]), np.array([0]), 
            vector[0], vector[1], angles='xy', scale_units='xy', scale=1
        )
        plt.text(vector[0], vector[1], label)
    plt.show()


df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist()
train_labels = np.array(to_number(df.v1))

max_vocab_size = 1000 
max_sequence_length = 20 
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(train_texts)

sequences = tokenizer.texts_to_sequences(train_texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

embedding_dim = 2
model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
model.fit(padded_sequences, train_labels, epochs=epochs, batch_size=32, validation_split=0.2)

selected_words = ['free', 'win', 'money', 'call', 'ham']
indices = [tokenizer.word_index[word] for word in selected_words if word in tokenizer.word_index]

plot_vectors(model, tokenizer, indices=indices)
