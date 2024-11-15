import os
import re
import pandas as pd
import spacy
from tqdm import tqdm

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    """Load data from a CSV or Excel file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

def anonymize_pii(text):
    """Anonymize PII in the text such as names, emails, and phone numbers."""
    # Anonymize email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    # Anonymize phone numbers
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)

    # Use spaCy NER to detect and replace names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, "<NAME>")
    return text

def clean_resume_text(text):
    """Clean and preprocess resume text."""
    text = anonymize_pii(text)  # Anonymize PII first
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove unwanted punctuation after anonymization
    # Remove unwanted spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()
    return text

def preprocess_data(df):
    """Preprocess the data to create 'Resume_str' and 'relevance' columns."""
    # Drop unnecessary columns if they exist
    df.drop(columns=[col for col in ['Resume_html', 'ID'] if col in df.columns], inplace=True, errors='ignore')

    # Handle 'Category' and create 'relevance' column
    if 'Category' in df.columns:
        df['relevance'] = df['Category'].apply(lambda x: 1 if x == 'INFORMATION-TECHNOLOGY' else 0)
        df.drop(columns='Category', inplace=True)
    elif 'relevance' not in df.columns:
        raise ValueError("Data must contain 'Category' or 'relevance' column.")

    # Identify resume text column
    text_columns = ['Resume_str', 'resume_text']
    resume_col = next((col for col in text_columns if col in df.columns), None)
    if not resume_col:
        raise ValueError("Data must contain 'Resume_str' or 'resume_text' column.")

    # Rename resume text column to 'Resume_str' if necessary
    if resume_col != 'Resume_str':
        df.rename(columns={resume_col: 'Resume_str'}, inplace=True)

    # Select only 'Resume_str' and 'relevance' columns for balancing
    return df[['Resume_str', 'relevance']]

def save_data(df, output_path):
    """Save the processed data to a CSV file."""
    df.to_csv(output_path, index=False)

def main():
    # File paths
    primary_data_path = 'Dataset/Resume/Resume.csv'
    additional_data_path = 'Dataset/Resume/resumes2.xlsx'
    output_path = 'processed_data.csv'

    # Load and preprocess the primary data
    primary_data = load_data(primary_data_path)
    processed_primary_data = preprocess_data(primary_data)

    # Load and preprocess the additional data
    additional_data = load_data(additional_data_path)
    additional_data['relevance'] = 1  # Mark all as relevant
    processed_additional_data = preprocess_data(additional_data)

    # Combine datasets and balance them by relevance
    combined_data = pd.concat([processed_primary_data, processed_additional_data], ignore_index=True)
    min_count = combined_data['relevance'].value_counts().min()
    balanced_data = combined_data.groupby('relevance', group_keys=False).apply(
        lambda x: x.sample(min_count, random_state=42)
    ).reset_index(drop=True)

    # Apply the clean_resume_text function to each resume with tqdm progress bar after balancing
    tqdm.pandas(desc="Cleaning resumes")
    balanced_data['Resume_str'] = balanced_data['Resume_str'].astype(str).progress_apply(clean_resume_text)

    # Ensure only the required columns are saved
    balanced_data = balanced_data[['Resume_str', 'relevance']]
    save_data(balanced_data, output_path)

if __name__ == "__main__":
    main()
