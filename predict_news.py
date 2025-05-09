import joblib

# Load the saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Get user input
news = input("Enter a news headline or article: ")

# Transform the input using the vectorizer
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Show result
if prediction[0] == 0:
    print("The news is Real.")
else:
    print("The news is Fake.")