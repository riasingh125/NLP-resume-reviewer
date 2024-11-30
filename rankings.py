import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def rank_relevant_resumes(model, vectorizer, data, job_description):
    """Rank relevant resumes based on similarity to the job description."""
    # Filter relevant resumes
    relevant_resumes = data[data["relevance"] == 1]["Resume_str"]
    relevant_embeddings = vectorizer.transform(relevant_resumes).toarray()

    # Embed job description
    job_description_embedding = vectorizer.transform([job_description]).toarray()

    # Reduce dimensionality
    svd = TruncatedSVD(n_components=100)
    reduced_embeddings = svd.fit_transform(relevant_embeddings)
    reduced_job_desc = svd.transform(job_description_embedding)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(reduced_job_desc, reduced_embeddings).flatten()

    # Rank resumes by similarity
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_resumes = [(similarity_scores[i], relevant_resumes.iloc[i]) for i in ranked_indices]
    return ranked_resumes
