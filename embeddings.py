import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Funktion für die Umwandlung der Labels
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
    
# CSV
df = pd.read_csv("spam.csv", encoding="latin-1")
train_texts = df.v2.tolist() # Textnachrichten
train_labels = np.array(to_number(df.v1)) # Labels: ham (0) und spam (1)

max_words = 10000
max_length = 100

tokenizer = Tokenizer(num_words=max_words).fit_on_texts(train_texts)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')


def main():
    print("Wähle eine Option:")
    print("1. Debugging-Tool: Vokabular anzeigen")
    print("2. Modell trainieren und Embeddings visualisieren")
    choice = input("Gib 1 oder 2 ein: ")

    if choice == "1":
        get_vocab_indices(tokenizer)
    elif choice == "2":
        # Modell erstellen
        embedding_dim = 2  # Dimension der Embeddings
        model = Sequential([
            Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # Komplieren
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Trainieren
        history = model.fit(
            padded_sequences,
            train_labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        # Embeddings visualisieren
        selected_words = ["spam", "ham", "Jackpot", "win", "money", "award"]  # Beispielwörter
        selected_indices = [tokenizer.word_index[word] 
                            for word in selected_words 
                            if word in tokenizer.word_index]

        plot_vectors(model, tokenizer, indices=selected_indices)
    else:
        print("Ungültige Eingabe. Bitte starte das Programm erneut und gib 1 oder 2 ein.")

# Programmstart
if __name__ == "__main__":
    main()