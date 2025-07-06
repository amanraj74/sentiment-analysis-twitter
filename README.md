# Sentiment Analysis on Twitter Data

A professional, easy-to-understand Python project that classifies tweets as positive, negative, or neutral using Natural Language Processing (NLP) and machine learning.

## Features
- Cleans and preprocesses tweet text
- Converts text to numerical features using TF-IDF
- Trains a Logistic Regression classifier
- Evaluates model performance with metrics and a confusion matrix
- Well-commented, modular, and human-readable code

## Dataset
A sample dataset (`tweets.csv`) is included. You can replace it with your own dataset (CSV with columns: `tweet`, `sentiment`).

## Setup Instructions
1. **Clone the repository or download the project files.**
2. **Install required Python libraries:**
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn
   ```
3. **Run the script:**
   ```bash
   python sentiment_analysis.py
   ```

## Usage
- The script will:
  - Load and preprocess the tweets
  - Extract features and train a model
  - Print evaluation metrics and show a confusion matrix
  - Display sample predictions

## Example Output
```
Sample data:
                                 tweet sentiment
0  I love the new design of your website!  positive
1     This is the worst service I have ever used.  negative
...

Preprocessing tweets...
...

Extracting features with TF-IDF...

Training Logistic Regression model...

Classification Report:
              precision    recall  f1-score   support
    negative       1.00      1.00      1.00         1
     neutral       1.00      1.00      1.00         1
    positive       1.00      1.00      1.00         1

...
```

## Customization
- To use your own data, replace `tweets.csv` with your dataset (same columns).
- You can expand the script to use other models or more data.

## Credits
- Developed by [AMAN JAISWAL]
- For CodeC Technologies assignment

---
Feel free to use, modify, and share this project! 