import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from wordcloud import WordCloud
import streamlit as st

# Load ranked resumes
ranked_resumes_file = 'ranked_resumes.csv'  # Replace with your file path
ranked_resumes = pd.read_csv(ranked_resumes_file)

# Streamlit Dashboard
st.title("Interactive Resume Visualization Dashboard")
st.sidebar.header("Dashboard Settings")

# Determine the maximum value dynamically based on the number of resumes
max_resumes = len(ranked_resumes)

# Sidebar Configurations
top_n = st.sidebar.slider(
    "Select Number of Top Resumes to Visualize:",
    min_value=1,
    max_value=max_resumes,
    value=min(10, max_resumes)  # Default to 10 or total resumes if less than 10
)
score_threshold = st.sidebar.slider("Final Score Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
visualization_type = st.sidebar.selectbox(
    "Choose Visualization:",
    ["Bar Chart", "Heatmap", "Word Cloud", "Score Distribution", "Detailed Resume Comparison"]
)

# Preprocessing Resumes for Display
ranked_resumes['Truncated_Resume'] = ranked_resumes['Resume_str'].apply(
    lambda x: x[:50] + '...' if len(x) > 50 else x
)

# Visualization 1: Bar Chart for Top Ranked Resumes
if visualization_type == "Bar Chart":
    st.header(f"Top {top_n} Ranked Resumes")
    top_resumes = ranked_resumes.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x='final_score',
        y='Truncated_Resume',
        data=top_resumes,
        palette='coolwarm',
        ax=ax
    )
    ax.set_xlabel('Final Score', fontsize=14)
    ax.set_ylabel('Resume (Truncated)', fontsize=14)
    ax.set_title(f'Top {top_n} Ranked Resumes', fontsize=16)
    st.pyplot(fig)

# Visualization 2: Heatmap of Score Components
elif visualization_type == "Heatmap":
    st.header(f"Heatmap of Score Components for Top {top_n} Resumes")

    required_columns = ['Truncated_Resume', 'keyword_score', 'similarity_score', 'final_score']
    missing_columns = [col for col in required_columns if col not in ranked_resumes.columns]

    if missing_columns:
        st.error(f"Missing columns for heatmap: {missing_columns}")
    else:
        top_resumes = ranked_resumes.head(top_n)
        heatmap_data = top_resumes[required_columns].set_index('Truncated_Resume')

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Score Components Heatmap", fontsize=16)
        st.pyplot(fig)

# Visualization 3: Word Cloud for Relevant Resumes
elif visualization_type == "Word Cloud":
    st.header("Word Cloud of Relevant Resumes")
    relevant_resumes = ranked_resumes[ranked_resumes['final_score'] >= score_threshold]
    if not relevant_resumes.empty:
        text = ' '.join(relevant_resumes['Resume_str'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud", fontsize=16)
        st.pyplot(fig)
    else:
        st.write("No resumes meet the score threshold for the Word Cloud.")

# Visualization 4: Score Distribution
elif visualization_type == "Score Distribution":
    st.header("Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ranked_resumes['final_score'], bins=20, kde=True, ax=ax, color='green')
    ax.set_xlabel('Final Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of Final Scores', fontsize=16)
    st.pyplot(fig)

# Visualization 5: Detailed Resume Comparison
elif visualization_type == "Detailed Resume Comparison":
    st.header("Detailed Resume Comparison")
    selected_resumes = st.multiselect(
        "Select Resumes for Detailed Comparison:",
        options=ranked_resumes['Resume_str'],
        default=ranked_resumes['Resume_str'].iloc[:2]
    )
    if selected_resumes:
        comparison_df = ranked_resumes[ranked_resumes['Resume_str'].isin(selected_resumes)][
            ['Resume_str', 'keyword_score', 'similarity_score', 'final_score']
        ]
        st.table(comparison_df)
    else:
        st.write("Select at least one resume for comparison.")

# Interactive Table
st.header("Interactive Resume Table")
filtered_resumes = ranked_resumes[ranked_resumes['final_score'] >= score_threshold]
st.dataframe(filtered_resumes)

# Download Button for Filtered Resumes
csv_download = filtered_resumes.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Resumes as CSV",
    data=csv_download,
    file_name="filtered_resumes.csv",
    mime="text/csv"
)


#Other Visualizations: Evaluation Metrics

#F1 score, Mean Reciprocol Rank, and Normlaized Discounted Cumulative Gain  
def plot_evaluation_metrics(f1, mrr, ndcg):
    """Plot F1, MRR, and nDCG as a bar chart."""
    metrics = {'F1 Score': f1, 'MRR': mrr, 'nDCG': ndcg}
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
    plt.title('Evaluation Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylim(0, 1)  # Scores are typically between 0 and 1
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')
    plt.savefig('evaluation_metrics.png')
    plt.show()
    
def plot_score_distribution(scores):
    """Plot the distribution of final scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Scores', fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig('score_distribution.png')
    plt.show()



