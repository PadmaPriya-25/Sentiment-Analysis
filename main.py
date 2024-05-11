import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('vader_lexicon')

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print("Error reading CSV file:", e)
        return None

def perform_sentiment_analysis(comments):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for comment in comments:
        if isinstance(comment, str):  # Check if comment is a string
            sentiment_score = sid.polarity_scores(comment)
            sentiment_scores.append(sentiment_score)
        else:
            sentiment_scores.append({'compound': 0.0})  # Assigning neutral score for non-string comments
    sentiment_scores_df = pd.DataFrame(sentiment_scores)
    return sentiment_scores_df

def visualize_sentiment_distribution(sentiment_scores_df):
    sentiment_counts = sentiment_scores_df.idxmax(axis=1).value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def main():
    # Step 1: User Input - File Path
    file_path = input("Enter the path of the CSV file: ")

    # Step 2: Read CSV File
    df = read_csv(file_path)
    if df is None:
        return

    # Step 3: Sentiment Analysis
    if 'COMMENT' not in df.columns:
        print("Error: 'COMMENT' column not found in the CSV file.")
        return
    sentiment_scores_df = perform_sentiment_analysis(df['COMMENT'])

    # Step 4: Visualization
    visualize_sentiment_distribution(sentiment_scores_df)

if __name__ == "__main__":
    main()
