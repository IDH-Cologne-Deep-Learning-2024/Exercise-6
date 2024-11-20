import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
df = pd.read_csv("spam.csv", encoding="latin-1")
texts, labels = df.v2.tolist(), df.v1.map({'ham': 0, 'spam': 1}).values

# Split data
np.random.seed(42)
indices = np.random.permutation(len(texts))
split_point = int(0.8 * len(texts))
X_train, y_train = [texts[i] for i in indices[:split_point]], labels[indices[:split_point]]
X_test, y_test = [texts[i] for i in indices[split_point:]], labels[indices[split_point:]]

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), padding='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=X_train_pad.shape[1], padding='post')

# Create and train model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 2, input_length=X_train_pad.shape[1]),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=10, validation_split=0.2, batch_size=32, verbose=0)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history and word embeddings
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

embeddings = model.layers[0].get_weights()[0]
words_to_plot = ['free', 'win', 'call', 'text', 'money']
word_indices = [tokenizer.word_index[word] for word in words_to_plot if word in tokenizer.word_index]
colors = plt.cm.rainbow(np.linspace(0, 1, len(word_indices)))

for (index, word), color in zip(zip(word_indices, words_to_plot), colors):
    vec = embeddings[index]
    ax3.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color, label=word)
    ax3.annotate(word, (vec[0], vec[1]))

ax3.set_title('Word Embeddings')
ax3.set_xlabel('Dimension 1')
ax3.set_ylabel('Dimension 2')
ax3.legend()
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()
