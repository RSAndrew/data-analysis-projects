import pandas as pd
import numpy as np
from textblob import TextBlob

# Generate dummy customer reviews data
np.random.seed(0)
num_reviews = 100
reviews = ["Excellent product!", "Terrible service.", "Average experience.", "Highly recommended.",
           "Not worth the price.", "Great value for money.", "Disappointed with the quality.",
           "Outstanding customer support.", "Poor packaging.", "Impressed by the features."]

sentiments = [TextBlob(review).sentiment.polarity for review in reviews]

# Create a DataFrame
data = {
    'Review': reviews,
    'Sentiment': sentiments
}
df = pd.DataFrame(data)

# Save the DataFrame to Excel
excel_file = 'customer_reviews.xlsx'
df.to_excel(excel_file, index=False)

# Analyze sentiments
positive_reviews = df[df['Sentiment'] > 0]
negative_reviews = df[df['Sentiment'] < 0]
neutral_reviews = df[df['Sentiment'] == 0]

print('Excel file "customer_reviews.xlsx" with dummy data generated.')
print('Number of Positive Reviews:', len(positive_reviews))
print('Number of Negative Reviews:', len(negative_reviews))
print('Number of Neutral Reviews:', len(neutral_reviews))
