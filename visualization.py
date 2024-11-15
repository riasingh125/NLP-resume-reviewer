import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

# Load ranked resumes
ranked_resumes_file = 'ranked_resumes.csv'  # Replace with your file path
ranked_resumes = pd.read_csv(ranked_resumes_file)


# Visualization 1: Bar Chart for Top Ranked Resumes
def bar_chart_top_resumes(ranked_resumes, top_n=10):
    """Visualize the top N resumes and their final scores."""
    top_resumes = ranked_resumes.head(top_n).copy()
    top_resumes['Resume'] = top_resumes['Resume_str'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

    plt.figure(figsize=(12, 6))
    plt.barh(top_resumes['Resume'], top_resumes['final_score'], color='skyblue')
    plt.xlabel('Final Score', fontsize=14)
    plt.ylabel('Resume (truncated)', fontsize=14)
    plt.title(f'Top {top_n} Ranked Resumes', fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.tight_layout()
    plt.show()


# Visualization 2: Heatmap of Score Components
def heatmap_score_components(ranked_resumes, top_n=10):
    """Visualize the components contributing to the final score for top resumes."""
    top_resumes = ranked_resumes.head(top_n).copy()
    top_resumes['Resume'] = top_resumes['Resume_str'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

    # Simulating Keyword Score and Similarity Score (if not already in the file)
    keywords = ['Python', 'Machine Learning', 'TensorFlow', 'PyTorch', 'Data Analysis']  # Example keywords

    def keyword_match_score(resume, keywords):
        resume_words = resume.lower().split()
        return sum(1 for word in resume_words if word.lower() in keywords) / len(keywords)

    def mock_similarity_score(resume):
        # Placeholder for similarity scores, replace with your actual similarity computation
        return len(resume) % 100 / 100.0

    top_resumes['Keyword Score'] = top_resumes['Resume_str'].apply(lambda x: keyword_match_score(x, keywords))
    top_resumes['Similarity Score'] = top_resumes['Resume_str'].apply(mock_similarity_score)
    top_resumes = top_resumes[['Resume', 'relevance_score', 'Keyword Score', 'Similarity Score', 'final_score']]
    top_resumes.set_index('Resume', inplace=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        top_resumes[['relevance_score', 'Keyword Score', 'Similarity Score', 'final_score']],
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )
    plt.title('Score Components for Top Resumes', fontsize=16)
    plt.show()


# Visualization 3: Word Cloud for Relevant Resumes
def word_cloud_relevant_resumes(ranked_resumes):
    """Generate a word cloud from relevant resumes."""
    relevant_resumes = ranked_resumes[ranked_resumes['final_score'] > 0.5]  # Threshold for relevance
    text = ' '.join(relevant_resumes['Resume_str'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Relevant Resumes', fontsize=16)
    plt.show()


# Execute visualizations
print("Generating Bar Chart...")
bar_chart_top_resumes(ranked_resumes)

print("Generating Heatmap of Score Components...")
heatmap_score_components(ranked_resumes)

print("Generating Word Cloud...")
word_cloud_relevant_resumes(ranked_resumes)
