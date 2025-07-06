# sentiment_analysis.py
# Install requirements:
# pip install pandas numpy scikit-learn matplotlib seaborn spacy
# python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import re
import string
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Set random seed for reproducibility
RANDOM_SEED = 42

# Path to the dataset
DATA_PATH = 'tweets.csv'

# Text preprocessing functions

def clean_text(text):
    """
    Clean tweet text by removing URLs, mentions, hashtags, numbers, and punctuation.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)   # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)   # Remove hashtags
    text = re.sub(r'\d+', '', text)             # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    return text

def preprocess_tweet(text):
    """
    Tokenize, remove stopwords using spaCy, and join tokens back to string.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def predict_custom_tweet(model, vectorizer):
    """
    Prompt the user for a tweet, preprocess it, and predict its sentiment.
    """
    tweet = input("\nEnter a tweet to analyze its sentiment: ")
    clean = clean_text(tweet)
    processed = preprocess_tweet(clean)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    print(f"Predicted sentiment: {prediction}")

def main():
    """
    Main function to run the sentiment analysis pipeline:
    1. Load and preprocess data
    2. Extract features
    3. Train and evaluate model
    4. Predict custom tweet sentiment
    """
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    print('Sample data:')
    print(df.head())

    # Preprocess tweets
    print('\nPreprocessing tweets...')
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df['processed_tweet'] = df['clean_tweet'].apply(preprocess_tweet)
    print(df[['tweet', 'processed_tweet', 'sentiment']].head())

    # Feature extraction using TF-IDF
    print('\nExtracting features with TF-IDF...')
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_tweet'])
    y = df['sentiment']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Train Logistic Regression model
    print('\nTraining Logistic Regression model...')
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Show some predictions
    print('\nSample predictions:')
    sample_results = pd.DataFrame({'Tweet': df['tweet'].iloc[y_test.index], 'Actual': y_test, 'Predicted': y_pred})
    print(sample_results.head())

    # Predict sentiment for a custom tweet
    predict_custom_tweet(model, vectorizer)

if __name__ == "__main__":
    main() 