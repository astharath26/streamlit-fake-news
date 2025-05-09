import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the fake and real news data
fake_df = pd.read_csv("Fake.csv.csv")
true_df = pd.read_csv("True.csv.csv")

# Add labels: 1 for fake, 0 for real
fake_df["label"] = 1
true_df["label"] = 0

# Combine the datasets
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42)

# Separate features (X) and target (y)
# Assuming your text data is in a column named 'text'
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer for text vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(">> Training Model Now...")

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")