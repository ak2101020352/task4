#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure plotting
plt.style.use('seaborn')
# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Display the first few rows of the dataset
print(df.head())
# Basic statistics and information about the dataset
print(df.describe())
print(df.info())

# Check the distribution of the target variable
print(df['v1'].value_counts())
sns.countplot(x='v1', data=df)
plt.title('Distribution of Spam and Ham Emails')
plt.show()
# Rename columns for easier access
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Encode the target variable
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check for missing values
print(df.isnull().sum())

# Preprocessing text data
X = df['message']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Example: Predict if a new email is spam or not
new_email = ["Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now."]
new_email_tfidf = vectorizer.transform(new_email)
prediction = model.predict(new_email_tfidf)
print(f'Predicted label: {"spam" if prediction[0] == 1 else "ham"}')


# In[ ]:




