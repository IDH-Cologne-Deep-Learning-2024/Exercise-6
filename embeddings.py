import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Convert labels to numerical format
def to_number(labels):
    return [1 if label == 'spam' else 0 for label in labels]

# Plot vectors for selected words
def plot_vectors(model, tokenizer, words):
    embeddings = model.layers[0].get_weights()[0]
    word_indices = [tokenizer.word_index[word] for word in words if word in tokenizer.word_index]
    
    for index in word_indices:
        vector = embeddings[index]
        plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue')
        plt.text(vector[0], vector[1], list(tokenizer.word_index.keys())[index - 1], fontsize=12)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Word Embeddings Visualization')
    plt.grid(True)
    plt.show()

# Load the CSV file and preprocess data
df = pd.read_csv('spam.csv', encoding='latin-1')
texts = df['v2'].tolist()
labels = np.array(to_number(df['v1']))

# Tokenize and pad the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=2, input_length=max_length),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Visualize word embeddings for selected words
selected_words = ['free', 'win', 'urgent', 'hello', 'love']
plot_vectors(model, tokenizer, selected_words)
