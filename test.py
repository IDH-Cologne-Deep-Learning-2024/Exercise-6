import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
import matplotlib.pyplot as plt

# Load data
file_path = 'spam.csv'  # Replace with the path to your spam.csv file
data = pd.read_csv(file_path, encoding="latin-1")
data = data[['v1', 'v2']]  # Keep only the relevant columns
data.columns = ['label', 'message']

# Convert labels to numerical values
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Tokenize the messages
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['message'])
sequences = tokenizer.texts_to_sequences(data['message'])
word_index = tokenizer.word_index

# Pad sequences
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Prepare data for training
X = padded_sequences
y = data['label_num'].values

# Build the model
embedding_dim = 2
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_len),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Visualization Function
def plot_vectors(model, tokenizer, words):
    embeddings = model.layers[0].get_weights()[0]
    word_indices = [tokenizer.word_index[word] for word in words if word in tokenizer.word_index]
    vectors = embeddings[word_indices]
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        if word in tokenizer.word_index:
            plt.scatter(vectors[i, 0], vectors[i, 1])
            plt.text(vectors[i, 0] + 0.01, vectors[i, 1] + 0.01, word, fontsize=9)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title("Word Embeddings Visualization")
    plt.grid()
    plt.show()

# Visualize selected words
selected_words = ['free', 'win', 'cash', 'urgent', 'call']  # Add words you are interested in
plot_vectors(model, tokenizer, selected_words)
