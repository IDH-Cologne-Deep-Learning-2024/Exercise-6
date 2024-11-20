import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")
df_cleaned = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df_cleaned['label'] = df_cleaned['label'].map({'ham': 0, 'spam': 1})

texts = df_cleaned['message'].tolist()
labels = df_cleaned['label'].values

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

embedding_dim = 2 
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, name="embedding"),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32)

def plot_vectors(model, tokenizer, words):
    embedding_layer = model.get_layer("embedding")
    weights = embedding_layer.get_weights()[0]

    word_indices = [tokenizer.word_index[word] for word in words if word in tokenizer.word_index]
    selected_embeddings = weights[word_indices]

    plt.figure(figsize=(10, 10))
    for word, vector in zip(words, selected_embeddings):
        plt.scatter(vector[0], vector[1])
        plt.annotate(word, xy=(vector[0], vector[1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.title("Word Embeddings")
    plt.show()

# visualize
selected_words = ['free', 'win', 'call', 'love', 'money'] 
plot_vectors(model, tokenizer, selected_words)