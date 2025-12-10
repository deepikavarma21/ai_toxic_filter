import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load the data
train_df = pd.read_csv('datasets/train.csv')

# 2. Remove rows with missing text or labels
train_df = train_df.dropna(subset=['comment_text', 'toxic'])

# 3. Features and labels
X = train_df['comment_text']
y = train_df['toxic']

# 4. Split for validation (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorize text
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 6. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 7. Save vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('toxicity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 8. Optional: check accuracy on validation set
val_score = model.score(X_val_vec, y_val)
print(f'Validation Accuracy: {val_score:.2f}')
