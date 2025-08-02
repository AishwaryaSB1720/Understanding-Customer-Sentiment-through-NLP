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

# Map Ratings to Sentiment Labels
def map_sentiment(rating):
    if rating in [4, 5]:
        return 1  # Positive
    elif rating in [1, 2]:
        return 0  # Negative
    else:
        return 2  # Neutral
df['Sentiment'] = df['Rating'].apply(map_sentiment)
df['Sentiment'].value_counts()

# Upsample All Classes to Match the Largest Class
from sklearn.utils import resample
df_0 = df[df['Sentiment'] == 0]
df_1 = df[df['Sentiment'] == 1]
df_2 = df[df['Sentiment'] == 2]
max_samples = max(len(df_0), len(df_1), len(df_2))
df_0_up = resample(df_0, replace=True, n_samples=max_samples, random_state=42)
df_1_up = resample(df_1, replace=True, n_samples=max_samples, random_state=42)
df_2_up = resample(df_2, replace=True, n_samples=max_samples, random_state=42)
df_balanced = pd.concat([df_0_up, df_1_up, df_2_up]).sample(frac=1, random_state=42)

# Vectorization with TF-IDF
X = df_balanced['cleaned_review']
y = df_balanced['Sentiment']
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Apply models , Logistic Regression, Random Forest, XgBoost 
lr_model = LogisticRegression(max_iter=300, multi_class='multinomial', random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
y_pred_lr = lr_model.predict(X_train)
print("Training Accuracy Score:", accuracy_score(y_train, y_pred_lr))

rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
y_pred_rf = rf_model.predict(X_train)
print("Training Accuracy Score:", accuracy_score(y_train, y_pred_rf))

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\n XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
y_pred_xgb = xgb_model.predict(X_train)
print("Training Accuracy Score:", accuracy_score(y_train, y_pred_xgb))

# Here are some snapshots of the output of ML models

<img width="817" height="603" alt="Logistic Regression" src="https://github.com/user-attachments/assets/5087ac24-2c1b-4e65-9d5c-49cbc6a8ca00" />
<img width="977" height="622" alt="Random Forest" src="https://github.com/user-attachments/assets/43ddbaba-e29c-45d8-97fb-b23e33664888" />
<img width="977" height="762" alt="XgBoost " src="https://github.com/user-attachments/assets/6bc10d93-b7c4-492a-80dd-6c54c963eeaf" />
