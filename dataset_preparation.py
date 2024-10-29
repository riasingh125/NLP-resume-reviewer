import pandas as pd
import os
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """Load data from a CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def preprocess_data(df):
    """Preprocess the data."""
    print("Initial shape of the data:", df.shape)
    print(df[:5])
    print("len of df:", len(df))

    # print all the columns
    print("Columns in the data:")
    print(df.columns)

    # drop the html column
    df.drop(columns=['Resume_html'], inplace=True)

    # print all categories
    print("Categories in the data:")
    print(df['Category'].unique())

    # Create a new column 'relevance' based on the 'Category' column
    # 0 means not relevant, 1 means relevant (here we consider 'INFORMATION-TECHNOLOGY' as relevant)
    df['relevance'] = df['Category'].apply(lambda x: 1 if x == 'INFORMATION-TECHNOLOGY' else 0)

    # for the column "Resume_str", we will preprocess the text by removing special characters and converting to lowercase
    df['Resume_str'] = df['Resume_str'].str.replace('[^a-zA-Z0-9 ]', '', regex=True).str.lower()

    # now lets remove stop words
    df['Resume_str'] = df['Resume_str'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))

    # drop category column
    df.drop(columns=['Category'], inplace=True)

    return df

def save_data(df, output_path):
    """Save the processed data to a CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage
    file_path = 'Dataset/Resume/Resume.csv'
    output_path = 'processed_data.csv'

    data = load_data(file_path)
    processed_data = preprocess_data(data)
    save_data(processed_data, output_path)