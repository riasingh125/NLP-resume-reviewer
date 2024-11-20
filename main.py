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
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained neural network model
nn_model_path = 'best_model.keras'
nn_model = load_model(nn_model_path)

# Define constants
max_words = 20000  # Limit vocabulary size for tokenizer
max_len = 4000  # Maximum sequence length for padding
relevance_threshold = 0.5  # Threshold for filtering resumes
weight_keywords = 0.6  # Redistributed weight
weight_similarity = 0.4  # Redistributed weight

# Preprocessing: Remove stopwords and lemmatize text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Job Description
job_description = """
Job Title: Generative AI Specialist / Engineer
Location: Remote / Hybrid / Onsite (Specify Location)

About Us:
At [Your Company Name], we are pioneers in artificial intelligence innovation, delivering cutting-edge generative AI solutions to transform industries. Our mission is to leverage state-of-the-art AI models to tackle complex challenges, drive innovation, and create meaningful value for our clients. Join a passionate team of AI enthusiasts, researchers, and engineers shaping the future of technology.

Role Overview:
We are seeking a talented and motivated Generative AI Specialist to join our team. In this role, you will design, develop, and deploy generative AI solutions across diverse applications such as natural language processing, text generation, image synthesis, and more. You will collaborate with cross-functional teams to implement advanced AI techniques and deliver innovative solutions.

Key Responsibilities:
- **Model Development**:  
  - Design, fine-tune, and deploy generative AI models (e.g., GPT, Stable Diffusion, DALL-E) tailored to specific use cases.  
  - Optimize models for accuracy, efficiency, and scalability.  

- **Application Integration**:  
  - Integrate generative AI capabilities into existing systems and develop new AI-powered products, such as chatbots, content generation tools, and recommendation systems.  
  - Collaborate with product and engineering teams to deliver seamless AI solutions.  

- **Data Engineering**:  
  - Collect, preprocess, and curate large datasets for training and fine-tuning generative AI models.  
  - Implement robust data pipelines to streamline AI workflows.  

- **Research and Innovation**:  
  - Stay updated with the latest advancements in generative AI, machine learning, and deep learning.  
  - Propose and implement innovative solutions using state-of-the-art techniques.  

- **Evaluation and Testing**:  
  - Develop metrics to evaluate model outputs and ensure alignment with business goals and ethical standards.  
  - Conduct thorough testing to identify and mitigate biases or limitations.  

- **Documentation and Collaboration**:  
  - Document model architectures, system designs, and research findings.  
  - Work closely with product managers, data scientists, and software engineers to align AI solutions with organizational goals.  

Required Qualifications:
- Bachelor’s or Master’s degree in Computer Science, Artificial Intelligence, Machine Learning, or a related field.  
- 2+ years of experience working with generative AI models (e.g., GPT, GANs, VAEs).  
- Proficiency in Python and frameworks such as PyTorch, TensorFlow, or Hugging Face Transformers.  
- Experience in fine-tuning pre-trained models for custom datasets.  
- Strong understanding of machine learning concepts, natural language processing (NLP), or computer vision.  

Preferred Qualifications:
- Experience with Reinforcement Learning from Human Feedback (RLHF).  
- Familiarity with vector databases and Retrieval-Augmented Generation (RAG) systems.  
- Knowledge of distributed training and deployment techniques.  
- Background in ethical AI practices, including fairness, transparency, and privacy.  

Soft Skills:
- Strong problem-solving skills and creativity in developing AI applications.  
- Effective communication skills to convey technical concepts to non-technical stakeholders.  
- A collaborative mindset with the ability to work in dynamic, fast-paced environments.  
"""

# Preprocess the job description
logging.info("Preprocessing job description...")
job_description = preprocess_text(job_description)

# Load resumes
file_path = 'test_data.csv'
resume_data = pd.read_csv(file_path)

# Ensure all entries are strings and handle missing values
resume_data['Resume_str'] = resume_data['Resume_str'].fillna('').astype(str)

# Tokenize and pad the resume strings
logging.info("Tokenizing and padding resumes...")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(resume_data['Resume_str'])
sequences = tokenizer.texts_to_sequences(resume_data['Resume_str'])
X = pad_sequences(sequences, maxlen=max_len)

# Predict relevance using the neural network
logging.info("Filtering relevant resumes using neural network...")
predictions = nn_model.predict(X)
resume_data['relevance_score'] = predictions.flatten()

# Filter relevant resumes (threshold = 0.5)
relevant_resumes = resume_data[resume_data['relevance_score'] >= relevance_threshold].copy()
logging.info(f"Number of relevant resumes: {len(relevant_resumes)}")

# Extract keywords from job description using TF-IDF
def extract_keywords(text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

logging.info("Extracting keywords from job description...")
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
    resume_words = set(resume.lower().split())
    return len(resume_words.intersection(keywords)) / len(keywords)

# Semantic similarity score using Sentence Transformers
logging.info("Loading Sentence-BERT model for semantic similarity...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
jd_embedding = sbert_model.encode(job_description)

def semantic_similarity_score(resume, jd_embedding, model):
    resume_embedding = model.encode(resume)
    similarity = cosine_similarity([jd_embedding], [resume_embedding])
    return similarity[0][0]

# Normalize scores
scaler = MinMaxScaler()

def normalize_scores(df, columns):
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])
    return df

logging.info("Scoring resumes...")
scores = []
for _, row in tqdm(relevant_resumes.iterrows(), total=len(relevant_resumes)):
    resume_text = row['Resume_str']

    # Keyword match score
    keyword_score = keyword_match_score(resume_text, expanded_keywords)

    # Semantic similarity score
    similarity_score = semantic_similarity_score(resume_text, jd_embedding, sbert_model)

    # Final weighted score (without relevance score)
    final_score = (
        weight_keywords * keyword_score
        + weight_similarity * similarity_score
    )
    scores.append({
        "Resume_str": resume_text,
        "keyword_score": keyword_score,
        "similarity_score": similarity_score,
        "final_score": final_score,
    })

# Convert scores to DataFrame
scores_df = pd.DataFrame(scores)

# Normalize scores
scores_df = normalize_scores(scores_df, ['keyword_score', 'similarity_score', 'final_score'])

# Sort by final score
ranked_resumes = scores_df.sort_values(by='final_score', ascending=False)

# Save the ranked resumes
output_file = 'ranked_resumes.csv'
ranked_resumes.to_csv(output_file, index=False)
logging.info(f"Ranked resumes saved to {output_file}")

# Visualization
sns.histplot(ranked_resumes['final_score'], kde=True)
plt.title('Final Score Distribution')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.show()
