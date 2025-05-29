import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Import your clean_text function
# from preprocess import clean_text
def clean_text(text):
    import re
    import string
    import unicodedata
    text = unicodedata.normalize("NFKC", text)  # Normalize unicode
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset
df = pd.read_csv("multilang_realistic_90000.csv")

# Optional: Remove duplicates
drop_duplicates = True
if drop_duplicates:
    before = df.shape[0]
    df.drop_duplicates(subset='message', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Removed {before - df.shape[0]} duplicate messages.")

# Map labels
df['label'] = df['label'].map({'ham': 'safe', 'spam': 'spam', 'fraud': 'fraud', 'scam': 'scam'})

# Clean messages
df['message'] = df['message'].apply(clean_text)

# Show label distribution
print("Label distribution:\n", df['label'].value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# If you want to use custom stopwords, you can modify the TfidfVectorizer
# Load custom stopwords from a file

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

custom_stopwords = load_stopwords('stopwords.txt') # Load from file

# Convert the set of stopwords to a list
custom_stopwords_list = list(custom_stopwords) # Add this line

# Build TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=custom_stopwords_list, ngram_range=(1, 2), max_df=0.95, min_df=2)),
    ('clf', MultinomialNB())
])

# Hyperparameter tuning for alpha
param_grid = {'clf__alpha': [0.1, 0.5, 1.0, 2.0]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best alpha value:", grid.best_params_['clf__alpha'])
print("Best cross-validation accuracy:", grid.best_score_)

# Predict using best estimator
y_pred = grid.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=grid.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=grid.classes_, yticklabels=grid.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the model using pickle
with open('sms_spam_classifier.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

# Also save with joblib
joblib.dump(grid.best_estimator_, 'sms_spam_classifier.joblib')

print("Model training completed and saved as 'sms_spam_classifier.pkl' and 'sms_spam_classifier.joblib'")
