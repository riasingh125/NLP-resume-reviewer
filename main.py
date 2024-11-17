import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained neural network model
nn_model_path = 'best_model.keras'
nn_model = load_model(nn_model_path)

# Define constants
max_words = 60000
max_len = 4000

# Load job description and resumes
job_description = """Job Title: Generative AI Specialist / Engineer
Location: Remote / Hybrid / Onsite (Specify Location)
About Us:
At [Your Company Name], we are at the forefront of artificial intelligence innovation, creating solutions that leverage generative AI to revolutionize industries. Our mission is to harness the power of cutting-edge AI models to solve complex challenges, drive innovation, and create value for our customers. Join a team of AI enthusiasts, researchers, and engineers who are passionate about building the future.

Role Overview:
We are seeking a talented and motivated Generative AI Specialist to join our team. In this role, you will work on designing, developing, and deploying generative AI solutions across diverse applications such as natural language understanding, text generation, image synthesis, and more. You will collaborate with cross-functional teams to implement state-of-the-art AI techniques that push the boundaries of what’s possible.

Key Responsibilities:
Model Development and Optimization:

Design, fine-tune, and deploy generative models (e.g., GPT, Stable Diffusion, DALL-E, or similar) for specific use cases.
Evaluate and optimize model performance for accuracy, efficiency, and scalability.
Application Development:

Integrate generative AI capabilities into existing systems or develop new products, such as chatbots, content generation tools, recommendation engines, or creative applications.
Collaborate with product and engineering teams to deliver AI-powered features.
Data Engineering and Preprocessing:

Collect, preprocess, and curate large datasets to train and fine-tune generative models.
Implement robust data pipelines to ensure seamless integration of models with applications.
Research and Innovation:

Stay updated with the latest advancements in generative AI, machine learning, and deep learning.
Propose and implement innovative solutions leveraging state-of-the-art generative AI techniques.
Evaluation and Testing:

Develop metrics and frameworks for evaluating model outputs, ensuring alignment with business goals and ethical standards.
Conduct rigorous testing to identify and mitigate potential biases or failure modes.
Documentation and Collaboration:

Document research findings, model architectures, and system designs.
Collaborate with cross-functional teams, including product managers, data scientists, and software engineers, to align generative AI solutions with organizational goals.
Required Qualifications:
Bachelor's or Master's degree in Computer Science, Artificial Intelligence, Machine Learning, or related field.
2+ years of experience working with generative AI models (e.g., GPT, Transformer-based architectures, GANs, VAEs).
Proficiency in Python and libraries such as PyTorch, TensorFlow, or Hugging Face Transformers.
Experience with fine-tuning pre-trained models on custom datasets.
Strong understanding of machine learning concepts, natural language processing (NLP), or computer vision.
Familiarity with cloud platforms (e.g., AWS, GCP, Azure) for AI model deployment.
Preferred Qualifications:
Experience with Reinforcement Learning from Human Feedback (RLHF).
Familiarity with vector databases and Retrieval-Augmented Generation (RAG) systems.
Knowledge of large-scale distributed training and deployment techniques.
Background in developing ethical AI systems with a focus on fairness, transparency, and privacy.
Soft Skills:
Excellent problem-solving skills and the ability to think creatively about AI applications.
Strong communication skills to articulate complex AI concepts to non-technical stakeholders.
A collaborative mindset and willingness to work in a dynamic, fast-paced environment.
"""

file_path = 'test_data.csv'
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

# Extract keywords from job description using TF-IDF
print("Extracting keywords from job description...")
def extract_keywords(text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

keywords = extract_keywords(job_description, top_n=20)

# Expand keywords using synonyms
def expand_keywords(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name().lower())
    return list(expanded_keywords)

expanded_keywords = expand_keywords(keywords)

def keyword_match_score(resume, keywords):
    resume_words = resume.lower().split()
    keyword_count = sum(1 for word in resume_words if word in keywords)
    return keyword_count / len(keywords)

# Semantic similarity score using Sentence Transformers
print("Loading Sentence-BERT model for semantic similarity...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
jd_embedding = sbert_model.encode(job_description)

def semantic_similarity_score(resume, jd_embedding, model):
    resume_embedding = model.encode(resume)
    similarity = cosine_similarity([jd_embedding], [resume_embedding])
    return similarity[0][0]

# Normalize scores
scaler = MinMaxScaler()

def normalize_scores(df, columns):
    df[columns] = scaler.fit_transform(df[columns])
    return df

print("Scoring resumes...")
scores = []
for _, row in tqdm(relevant_resumes.iterrows(), total=len(relevant_resumes)):
    resume_text = row['Resume_str']
    relevance_score = row['relevance_score']

    # Keyword match score
    keyword_score = keyword_match_score(resume_text, expanded_keywords)

    # Semantic similarity score
    similarity_score = semantic_similarity_score(resume_text, jd_embedding, sbert_model)

    # Final weighted score
    final_score = 0.4 * relevance_score + 0.3 * keyword_score + 0.3 * similarity_score
    scores.append({
        "Resume_str": resume_text,
        "relevance_score": relevance_score,
        "keyword_score": keyword_score,
        "similarity_score": similarity_score,
        "final_score": final_score,
    })

# Convert scores to DataFrame
scores_df = pd.DataFrame(scores)

# Normalize scores
scores_df = normalize_scores(scores_df, ['relevance_score', 'keyword_score', 'similarity_score', 'final_score'])

# Sort by final score
ranked_resumes = scores_df.sort_values(by='final_score', ascending=False)

# Save the ranked resumes
output_file = 'ranked_resumes.csv'
ranked_resumes.to_csv(output_file, index=False)
print(f"Ranked resumes saved to {output_file}")
