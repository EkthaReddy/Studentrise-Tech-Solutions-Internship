# Studentrise Tech Solutions Internship

This intership provides tasks realted to AIs. In those tasks we need to create the model for each assigned tasks.

<details>
  
<summary><h3>Spam Email Classifier</h3></summary>

Creating an AI model to classify emails as spam or not spam involves several steps, including data collection, preprocessing, feature extraction, and model training. Here's a step-by-step guide to building such a classifier:

**Step 1:** Data Collection
You can use the Enron email dataset, which is publicly available and commonly used for email classification tasks.

**Step 2:** Data Preprocessing
Preprocessing involves cleaning the text data to make it suitable for feature extraction and model training. Common preprocessing steps include:

Removing HTML tags: Emails often contain HTML content.

Removing special characters and numbers: These are usually not helpful for text classification.

Converting text to lowercase: This ensures uniformity.

Removing stop words: Words like "and", "the", etc., which are not useful for classification.

Stemming or lemmatization: Reducing words to their base forms.

**Step 3:** Feature Extraction
Feature extraction involves converting text data into numerical features that can be used by machine learning algorithms. Common techniques include:

Bag of Words (BoW): Representing text as a set of word frequencies.

Term Frequency-Inverse Document Frequency (TF-IDF): A more refined version of BoW that accounts for the importance of words.

Word Embeddings: Using pre-trained models like Word2Vec or GloVe to convert words into dense vectors.

**Step 4:** Model Training
Several machine learning algorithms can be used to train the classifier, including:

Logistic Regression: A simple and effective baseline model.
Naive Bayes: Particularly suited for text classification tasks.
Support Vector Machines (SVM): Effective for high-dimensional spaces.
Random Forest: An ensemble method that is robust and powerful.
Neural Networks: Including simple feedforward networks or more complex architectures like LSTM or Transformers for advanced performance.
**Step 5:** Model Evaluation
Evaluating the performance of your model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

**Step 6:** Deployment
Deploying the model in a production environment where it can classify incoming emails in real-time.

Here’s a simplified implementation using Python and common libraries like Scikit-learn, NLTK, and Pandas:

python
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the dataset (assuming it's already downloaded and extracted)
emails = pd.read_csv('enron_emails.csv')

# Preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
emails['clean_text'] = emails['message'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(emails['clean_text']).toarray()
y = emails['label']  # Assuming 'label' is the column with spam/ham labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

</details>

---
