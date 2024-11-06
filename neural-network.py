import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


file_path = 'processed_data.csv'
processed_data = pd.read_csv(file_path)
print(processed_data['Resume_str'])

non_string_entries = processed_data[~processed_data['Resume_str'].apply(lambda x: isinstance(x, str))]
print(f"Non-string entries:\n{non_string_entries}")

processed_data['Resume_str'] = processed_data['Resume_str'].fillna('').astype(str)

# Define parameters
max_words = 60000  # Vocabulary size for tokenizer
max_len = 4000  # Maximum length for padding sequences
embedding_dim = 100  # Dimensionality of Word2Vec embeddings

# Tokenize and pad the resume strings
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(processed_data['Resume_str'])
total_words = len(tokenizer.word_index)
print(f"Total unique words in dataset: {total_words}")
resume_lengths = [len(resume.split()) for resume in processed_data['Resume_str']]
avg_len = np.mean(resume_lengths)
max_len_in_data = np.max(resume_lengths)
print(f"Average resume length: {avg_len}")
print(f"Maximum resume length: {max_len_in_data}")
sequences = tokenizer.texts_to_sequences(processed_data['Resume_str'])
X = pad_sequences(sequences, maxlen=max_len)

# Prepare target variable (relevance)
y = processed_data['relevance'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Word2Vec on the resume data
resume_sentences = [resume.split() for resume in processed_data['Resume_str']]
word2vec_model = Word2Vec(resume_sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
word_vectors = word2vec_model.wv

# Create an embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]

# Build the CNN model
model = Sequential()

# Add embedding layer
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len, 
                    weights=[embedding_matrix], trainable=False))

# Add convolutional layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Add dropout for regularization
model.add(Dropout(0.5))

# Flatten the output
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Display model summary
model.summary()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict relevance for the first 10 samples in the test set
predictions = (model.predict(X_test[:10]) > 0.5).astype("int32")
print("Predictions:", predictions.flatten())
print("Actual:", y_test[:10])
