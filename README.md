# Exercise 6

This is the sixth exercise for the course [Deep Learning](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/wintersemester-2024-2025/deep-learning/).

Clone this repository to your local setup. Create a branch with the name of your GitHub account and switch to it.

In the repository, you find a CSV file with text messages and a label if they are spam or ham.

Train a feed-forward neural network with Keras (you can choose the number of layers, neurons, activation functions and epochs) to predict if a message is spam or ham.
To do this, tokenize the messages, pad all messages according to the longest text and include a trainable embedding layer in your network. The embedding layer should have 2 dimensions (usually embeddings will have around 300 dimensions, we are working with 2 here for visualization purposes).

After training the model, use the function `plot_vectors` in the file `embeddings.py` to look at selected words and how they appear in the vector space. Give your chosen words as list to the `words` argument of the function. You can get all words in the tokenizer via `tokenizer.word_index`.

Commit your changes and push them to the branch with the name of your GitHub account.

Deadline for this exercise is November 21, 2024, 08:00:00 CET.
