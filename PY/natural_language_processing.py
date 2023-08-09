import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Function to generate dummy text data
def generate_dummy_text_data(num_samples):
    np.random.seed(42)
    reviews = [
        'This product is great!',
        'I am not satisfied with the quality.',
        'Excellent service and fast delivery.',
        'Terrible experience, would not recommend.',
        'The food was delicious and well-prepared.'
    ]
    sentiment = np.random.choice(['Positive', 'Negative', 'Neutral'], size=num_samples)
    return pd.DataFrame({'Review': np.random.choice(reviews, size=num_samples), 'Sentiment': sentiment})

# Generate dummy text data
num_samples = 100
dummy_text_data = generate_dummy_text_data(num_samples)

# Save dummy text data to Excel file
excel_file = 'textdata.xlsx'
dummy_text_data.to_excel(excel_file, index=False)

# Load text data from Excel file
text_data = pd.read_excel(excel_file)

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

text_data['Cleaned_Text'] = text_data['Review'].apply(preprocess_text)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
text_data['VADER_Sentiment'] = text_data['Review'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] > 0 else 'Negative')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(text_data['Cleaned_Text'], text_data['Sentiment'], test_size=0.2, random_state=42)

# Text Classification using Naive Bayes
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
y_pred = classifier.predict(X_test_vectorized)

# Classification Report
classification_rep = classification_report(y_test, y_pred)

print(classification_rep)
