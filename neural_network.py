import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
import matplotlib.pyplot as plt


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text_with_bert(df, tokenizer, max_length=128):
    """Tokenize and prepare data for BERT."""
    return tokenizer(
        list(df['Resume_str']),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )

def train_bert_model(data):
    """Train a BERT model for classifying resumes."""

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Tokenize data
    tokenized_data = preprocess_text_with_bert(data, tokenizer)
    labels = tf.convert_to_tensor(data['relevance'].values, dtype=tf.int32)

    # Ensure labels are 1D
    labels = tf.squeeze(labels)

    # Split into training and validation sets
    train_size = int(0.8 * len(data))
    train_data = {key: val[:train_size] for key, val in tokenized_data.items()}
    val_data = {key: val[train_size:] for key, val in tokenized_data.items()}
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]
    print(train_labels.shape, train_labels.dtype)
    print(val_labels.shape, val_labels.dtype)


    # Debugging shapes
    print(train_labels.shape, train_labels.dtype)
    print(val_labels.shape, val_labels.dtype)
    #print(model.output)


    # Optimizer and learning rate schedule
    steps_per_epoch = len(train_data['input_ids']) // 16
    num_train_steps = steps_per_epoch * 5
    optimizer, _ = create_optimizer(
        init_lr=5e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=0.1 * num_train_steps,
        weight_decay_rate=0.01
    )

    # Compile the model with SparseCategoricalCrossentropy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        batch_size=16,
        epochs=5
    )

    '''
    lrs = [lr_schedule(i).numpy() for i in range(num_train_steps)]
    plt.plot(lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.show()
    '''

    
    return model, tokenizer


def predict_relevance_with_bert(model, tokenizer, test_data):
    """Predict relevance scores for test data using BERT."""
    # Tokenize test data
    tokenized_test_data = preprocess_text_with_bert(test_data, tokenizer)

    # Get predictions
    logits = model.predict(tokenized_test_data).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()

    # Return relevance scores (probability of being relevant)
    return probabilities[:, 1]  # Column 1 corresponds to class 1 (relevant)


'''

def preprocess_text(text):
    """Preprocess text for consistency and better model performance."""

    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

def train_model(data):
    """Train the CNN model for classifying resumes."""
    # Preprocess and split data
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["Resume_str"].apply(preprocess_text)).toarray()
    y = data["relevance"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define CNN model
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=X.shape[1]),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    return model, vectorizer

def predict_relevance(model, vectorizer, data):
    """Predict relevance scores for a dataset."""
    X = vectorizer.transform(data["Resume_str"].apply(preprocess_text)).toarray()
    predictions = model.predict(X).flatten()
    return predictions  # Return raw relevance scores (between 0 and 1)


'''
