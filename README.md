# Understanding-Customer-Sentiment-through-NLP
One of the leading woman clothing e-commerce companies would like to analyse the customer’s behaviour by analysing customer’s demographics and reviews submitted on the website. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import contractions

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Data
df = pd.read_csv('women clothing.csv', encoding='latin1')
df = df.dropna(subset=['Review Text']).reset_index(drop=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing Functions
def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    text = clean_text(text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w, 'v') for w in words if w not in stop_words]
    return ' '.join(words)

# Apply Preprocessing
df['cleaned_review'] = df['Review Text'].apply(preprocess)

# Word Frequency Analysis
all_words = ' '.join(df['cleaned_review']).split()
common_words = Counter(all_words).most_common(20)
word_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

# Visualize Common Words
plt.figure(figsize=(10, 5))
sns.barplot(data=word_df, x='Word', y='Frequency', palette='viridis')
plt.title("Top 20 Most Frequent Words in Reviews")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Topic Modeling using LDA
vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
X_topics = vectorizer.fit_transform(df['cleaned_review'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_topics)

for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx+1}:", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# Sentiment Labeling (Binary: Positive vs Negative)
df = df[df['Rating'].isin([1, 2, 4, 5])]
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['Sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression Classifier
clf_lr = LogisticRegression(max_iter=1000, solver='liblinear')
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy Score:", accuracy_score(y_test, y_pred_lr))

# Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))

# XGBoost Classifier
clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf_xgb.fit(X_train, y_train)
y_pred_xgb = clf_xgb.predict(X_test)

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy Score:", accuracy_score(y_test, y_pred_xgb))

<img width="666" height="418" alt="Random_Forest_accuracy" src="https://github.com/user-attachments/assets/1ee23841-1412-4378-b810-51a766274dab" />
<img width="765" height="543" alt="XGboost_accuracy" src="https://github.com/user-attachments/assets/8efd3aca-fcd0-4220-b9c9-956a62add1a3" />
<img width="583" height="442" alt="Logistic_Regression_accuracy" src="https://github.com/user-attachments/assets/2c78b3e5-c37c-4cee-b8a9-d719b93791d2" />
