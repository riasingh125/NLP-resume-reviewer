import pandas as pd
from neural_network import train_bert_model, predict_relevance_with_bert
from sklearn.metrics.pairwise import cosine_similarity

# File paths
processed_data_path = "processed_data.csv"
test_data_path = "test_data.csv"
output_path = "ranked_resumes.csv"

# Load datasets
processed_data = pd.read_csv(processed_data_path)
test_data = pd.read_csv(test_data_path)

# Train BERT model
print("Training BERT model...")
bert_model, tokenizer = train_bert_model(processed_data)

# Predict relevance for test data
print("Predicting relevance for test data with BERT...")
test_data['relevance_score'] = predict_relevance_with_bert(bert_model, tokenizer, test_data)

# Input job description
job_description = input("We are seeking a Cloud Solutions Architect with a proven track record in designing and implementing cloud infrastructure. The ideal candidate will have expertise in AWS, Azure, or Google Cloud Platform, along with strong problem-solving skills and a collaborative mindset. ")

# Tokenize and encode the job description
job_tokens = tokenizer(
    job_description,
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors="tf"
)
job_embedding = bert_model(job_tokens).logits.numpy()

# Compute similarity scores
test_embeddings = bert_model(test_data).logits.numpy()
similarity_scores = cosine_similarity(test_embeddings, job_embedding.reshape(1, -1)).flatten()

# Add similarity scores to the DataFrame
test_data['similarity_score'] = similarity_scores

# Sort by similarity score
test_data = test_data.sort_values(by="similarity_score", ascending=False)

# Save the ranked resumes
test_data.to_csv(output_path, index=False)
print(f"Ranked resumes with similarity scores saved to {output_path}.")
