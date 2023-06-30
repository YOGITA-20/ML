#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install panda


# In[12]:


pip install sklearn


# In[1]:


# Step 1: Gather the Data
import pandas as pd

# Load the dataset (replace 'dataset.csv' with the actual file path)

# data = open(r"C:\Users\hp\Documents\Zoom\emails.csv", "r", encoding="utf8")
data = pd.read_csv(r"C:\Users\hp\Documents\Zoom\mail_data.csv")



# Step 2: Preprocess the Data
# Perform preprocessing steps such as removing stopwords, tokenization, and converting text data into numerical features

# Step 3: Implement the Naive Bayes Algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Define the input features (X) and the target variable (y)
X = data['Message']  
y = data['Category']  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create TF-IDF vectorizer and transform the input text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 4: Evaluate the Model
# Make predictions on the testing dataset
y_pred = model.predict(X_test_vectorized)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score',f1)


# In[ ]:




