import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
score_threshold = st.sidebar.slider("Relevance Score Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
visualization_type = st.sidebar.selectbox(
    "Choose Visualization:",
    ["Bar Chart", "Heatmap", "Word Cloud", "Score Distribution", "Detailed Resume Comparison"]
)

# Visualization 1: Bar Chart for Top Ranked Resumes
if visualization_type == "Bar Chart":
    st.header(f"Top {top_n} Ranked Resumes")
    top_resumes = ranked_resumes.head(top_n).copy()
    top_resumes['Resume'] = top_resumes['Resume_str'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_resumes['Resume'], top_resumes['final_score'], color='skyblue')
    ax.set_xlabel('Final Score', fontsize=14)
    ax.set_ylabel('Resume (truncated)', fontsize=14)
    ax.set_title(f'Top {top_n} Ranked Resumes', fontsize=16)
    ax.invert_yaxis()  # Invert y-axis to have the highest score at the top
    st.pyplot(fig)

# Visualization 2: Heatmap of Score Components
elif visualization_type == "Heatmap":
    st.header(f"Heatmap of Score Components for Top {top_n} Resumes")
    top_resumes = ranked_resumes.head(top_n).copy()
    top_resumes['Resume'] = top_resumes['Resume_str'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    top_resumes = top_resumes[['Resume', 'relevance_score', 'keyword_score', 'similarity_score', 'final_score']].set_index('Resume')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        top_resumes,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        ax=ax
    )
    ax.set_title("Score Components Heatmap", fontsize=16)
    st.pyplot(fig)

# Visualization 3: Word Cloud for Relevant Resumes
elif visualization_type == "Word Cloud":
    st.header("Word Cloud of Relevant Resumes")
    relevant_resumes = ranked_resumes[ranked_resumes['final_score'] >= score_threshold]
    text = ' '.join(relevant_resumes['Resume_str'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Word Cloud", fontsize=16)
    st.pyplot(fig)

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
            ['Resume_str', 'relevance_score', 'keyword_score', 'similarity_score', 'final_score']
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
