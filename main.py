import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the trained neural network model
nn_model_path = 'best_model.keras'  # Path to the trained neural network model
nn_model = load_model(nn_model_path)

# Define constants
max_words = 60000  # Vocabulary size for tokenizer
max_len = 4000  # Maximum length for padding sequences

# Load job description and resumes
job_description = """Gen ai, rag, llm"""
file_path = 'new_resumes.csv'  # Path to the resumes CSV file
resume_data = pd.read_csv(file_path)

# Ensure all entries are strings and handle missing values
resume_data['Resume_str'] = resume_data['Resume_str'].fillna('').astype(str)

# Tokenize and pad the resume strings
print("Tokenizing resumes...")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(resume_data['Resume_str'])
sequences = tokenizer.texts_to_sequences(resume_data['Resume_str'])
X = pad_sequences(sequences, maxlen=max_len)

# Predict relevance using the neural network
print("Filtering relevant resumes...")
predictions = nn_model.predict(X)
resume_data['relevance_score'] = predictions.flatten()

# Filter relevant resumes (threshold = 0.5)
relevant_resumes = resume_data[resume_data['relevance_score'] >= 0.5].copy()

# Extract keywords from job description
print("Extracting keywords from job description...")
keywords = [word.lower() for word in job_description.split() if len(word) > 3]  # Simple keyword extraction


def keyword_match_score(resume, keywords):
    """Calculate the keyword match score for a resume."""
    resume_words = resume.lower().split()
    keyword_count = sum(1 for word in resume_words if word in keywords)
    return keyword_count / len(keywords)  # Normalize by the number of keywords


# Semantic similarity score using BERT embeddings
print("Loading BERT model for semantic similarity...")
bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)


def compute_bert_embeddings(text, tokenizer, model):
    """Generate BERT embeddings for a text."""
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=512)
    outputs = model(inputs.input_ids)
    return outputs.pooler_output.numpy()  # Use the pooled output for similarity


print("Computing job description embedding...")
jd_embedding = compute_bert_embeddings(job_description, bert_tokenizer, bert_model)


def semantic_similarity_score(resume, jd_embedding, tokenizer, model):
    """Calculate semantic similarity score between a resume and the job description."""
    resume_embedding = compute_bert_embeddings(resume, tokenizer, model)
    similarity = cosine_similarity(jd_embedding, resume_embedding)
    return similarity[0][0]  # Return the similarity score


# Scoring the relevant resumes
print("Scoring resumes...")
scores = []
for _, row in tqdm(relevant_resumes.iterrows(), total=len(relevant_resumes)):
    resume_text = row['Resume_str']
    relevance_score = row['relevance_score']

    # Keyword match score
    keyword_score = keyword_match_score(resume_text, keywords)

    # Semantic similarity score
    similarity_score = semantic_similarity_score(resume_text, jd_embedding, bert_tokenizer, bert_model)

    # Final combined score (weights can be adjusted)
    final_score = 0.4 * relevance_score + 0.3 * keyword_score + 0.3 * similarity_score
    scores.append(final_score)

relevant_resumes['final_score'] = scores

# Rank resumes by final score
ranked_resumes = relevant_resumes.sort_values(by='final_score', ascending=False)

# Save the ranked resumes
output_file = 'ranked_resumes.csv'
ranked_resumes.to_csv(output_file, index=False)
print(f"Ranked resumes saved to {output_file}")
