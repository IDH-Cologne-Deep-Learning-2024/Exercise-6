import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import svm 

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
sequences = tokenizer.texts_to_sequences(train_texts)
max_length = max(len(sequence) for sequence in sequences)
sequences_padded = pad_sequences(sequences, maxlen = max_length, padding= "post")

FFNN = Sequential()
FFNN.add(Embedding(vocab_size, 2, input_length= max_length))
FFNN.add(Flatten())
FFNN.add(Dense(7, activation="relu"))
FFNN.add(Dense(1, activation="sigmoid"))
FFNN.compile(loss='binary_crossentropy', optimizer="sgd")

FFNN.fit(x = sequences_padded, y = train_labels, epochs = 187, verbose = 2, validation_freq = 3)

scam_test_words = [2834, 226, 2744, 47, 318, 8102] #Rich, Money, Gang, Free, Car, Grandma
plot_vectors(FFNN, tokenizer, scam_test_words)

FFNN.summary()

#Money and grandma are closer than money and rich lul






"""
z = df['v2']
y = df["v1"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))
"""
