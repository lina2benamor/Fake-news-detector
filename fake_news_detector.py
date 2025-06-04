import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# Note: You'll need to download the dataset first
# Dataset can be found at: https://www.kaggle.com/c/fake-news/data
try:
    df = pd.read_csv('fake-news.csv')  # Update with your file path
except FileNotFoundError:
    print("Dataset not found. Please download it from Kaggle and update the path.")
    exit()

# Display basic info
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Step 2: Prepare the data
X = df['text']  # Features (text content)
y = df['label']  # Labels (0=real, 1=fake)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Step 4: Initialize and train the classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Step 5: Make predictions and evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {round(score*100, 2)}%')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
print('\nConfusion Matrix:')
print(conf_matrix)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 6: Save the model for future use
import joblib

# Save the vectorizer and model
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(pac, 'fake_news_classifier.joblib')

print("\nModel and vectorizer saved to disk.")

# Step 7: Example of using the saved model
def predict_news(text):
    # Load the vectorizer and model
    loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    loaded_model = joblib.load('fake_news_classifier.joblib')
    
    # Vectorize the input text
    text_vectorized = loaded_vectorizer.transform([text])
    
    # Make prediction
    prediction = loaded_model.predict(text_vectorized)
    
    return "Fake News" if prediction[0] == 1 else "Real News"

# Test with a sample text
sample_text = """
The president announced a new policy that will change the way we think about healthcare.
This is a groundbreaking initiative that will benefit millions of Americans.
"""
print("\nSample prediction:")
print(predict_news(sample_text))