# NLP-resume-reviewer
CS6120 Project Proposal
Group 11: Dhruv Kamalesh Kumar, Ria Singh

Basic Idea

We want to implement an NLP-powered resume reviewer that will take in a description of the job and a database of resumes. It will first filter our irrelevant resumes, and then automatically rank each resume based on its relevance to the given input. Ultimately, the hope is to save time for recruiters, admissions counselors, and hiring managers by showing the most qualified candidates. 

Approach to Solution

Part 1: Training the neural network to filter out irrelevant resumes:
The objective is to train a classifier neural network to remove the irrelevant resumes, by classifying each resume as relevant or not relevant, based on the input job description.
We would have a labeled dataset with resumes with “relevant” or “irrelevant” for the training process. Since labeled data is typically limited, the project may rely on self-labeled data or a synthetic dataset curated for supervised learning.
We will be using Word2Vec to create the resume and job description word embeddings and then pass them into a neural network like LSTM or CNN
We will train the neural network using binary cross-entropy for the prediction

Part 2: Ranking the Relevant Resumes:
The objective is to further sort the resumes, by ranking them based on their relevance to the specified job description
For text preprocessing, we will be using at the very least the NLTK library for tokenization, stopword removal, lemmatization, and vectorization
For the similarity calculation, we will be using either cosine similarity or Jaccard similarity
We will implement a ranking model based on the similarity scores 
We will use SVD for dimensionality reduction and, after further research, Lasso or Ridge to prevent overfitting. 
Related Work
Deshmukh, A., & Raut, A. (2024). Applying BERT-Based NLP for Automated Resume Screening and Candidate Ranking. Annals of Data Science.
This study’s relevance to our project lies in its exploration of BERT-based models for semantic similarity, which are effective in capturing the context and nuanced relationships within text. Given that BERT (Bidirectional Encoder Representations from Transformers) can understand the context of skills, experience, and qualifications in resumes relative to job descriptions, it serves as a benchmark for contextual embeddings. This research illustrates that BERT is particularly strong at aligning resumes to job requirements, demonstrating how pre-trained language models outperform traditional NLP methods in capturing semantic meaning. However, it also notes the increased computational load of BERT, informing our decision to consider lighter models, such as Word2Vec, for our initial approach. Studying BERT's performance and computational requirements provides insights into potentially integrating advanced embeddings if high computational resources are available or for the future scalability of our project. Available at: https://doi.org/10.1007/s40745-024-00524-5
Gupta, P., Sharma, R., & Malhotra, A. (2022). Automated Resume Filtering Using Hybrid NLP Techniques.
This work compares classical NLP techniques like TF-IDF with neural embeddings, making it particularly relevant to our hybrid approach of combining traditional NLP methods with neural networks for relevance ranking. Their findings show that neural embeddings improve accuracy in skill matching and candidate ranking by capturing relationships in the data that simpler models may overlook. By examining these hybrid techniques, we gain insights into balancing computational efficiency with accuracy. This study supports our choice to experiment with similarity metrics like cosine similarity and Jaccard similarity, which are classical NLP methods, alongside neural network-based techniques for further fine-tuning. Additionally, their analysis of combining various NLP methods for resume filtering helps guide our exploration of different preprocessing and feature extraction approaches, as well as our decision to integrate regularization to control model complexity.
GitHub Repository: https://github.com/prateekguptaiiitk/Resume_Filtering
Assessment Methodology
Performance Evaluation:
We will assess our model’s accuracy using metrics such as precision, recall, and F1 score to ensure it correctly classifies and ranks resumes.
Ranking Accuracy: Evaluating the ranked list of resumes in terms of Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (nDCG) to measure ranking relevance.
Cross-Validation Strategy:
k-Fold Cross-Validation: A 5-fold cross-validation strategy will be used to validate the model’s generalizability, ensuring consistent performance across various data subsets.
Ablation Settings:
Input Dimensions: Testing the effect of input dimensions by adjusting the embedding size or the resume length (e.g., summary vs. full resume) to observe how detail impacts ranking accuracy.
Preprocessing Techniques: Varying preprocessing techniques (e.g., including/excluding lemmatization or stopword removal) and measuring their effect on accuracy.
Algorithm Complexity: Experimenting with different model complexities, such as simpler algorithms like logistic regression versus neural networks, and comparing their performance impacts.
Timeline
Week-by-Week Breakdown
Week 1: Collect and preprocess the dataset, clean the data, and remove duplicates.
Week 2: Implement and train the traditional NLP-based ranking model, focusing on feature extraction and similarity scoring.
Week 3: Evaluate the approach, measure performance on defined metrics, and visualize results.
Week 4: Tweak functionality, aiming to achieve the highest accuracy and robustness. 
Final Week: Write the report, and conduct error analysis.
Responsibilities
Since this project is a collaborative effort, responsibilities are distributed as follows:
Dhruv Kamalesh Kumar: Responsible for setting up and testing the neural network, and evaluations.
Ria Singh: Focused on implementing the traditional NLP-based ranking model, including feature extraction and cosine similarity ranking with data preprocessing, and the evaluations for the ranking system. 

