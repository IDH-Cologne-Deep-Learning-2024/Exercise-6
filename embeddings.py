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

#tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)


# input matrix
vocab_size =  len(tokenizer.word_index) +1

# setting matrix dimension
embedding_dim = 2

# setting max size for embeddings
max_length = max(map(len, train_sequences)) # using map to become faster

#padding
tokenized_texts = pad_sequences(train_sequences, maxlen=max_length, padding='post')

# neural network
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_length)) #embedding layer
model.add(Flatten()) #making one dimesional 
model.add(Dense(4, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="sgd")

model.fit(x = tokenized_texts, y = train_labels, epochs = 100, verbose = 2, validation_freq = 3)

# input array for prediction
#input_array = ["This is an offer "]

# prediction
#output_array = model.predict(input_array)
#print(output_array.shape)

plot_vectors(model, tokenizer)

