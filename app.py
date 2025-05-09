import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Title and subtitle
st.markdown(
    "<h1 style='text-align: center; color: #003366;'>Fake News Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Enter a news statement to check whether it's real or fake.</p>",
    unsafe_allow_html=True
)

# Input box
news_text = st.text_area("Enter News Text:", height=200)

# Button
if st.button("Check"):
    if news_text.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        # Vectorize and predict
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.error("Result: Fake News")
        else:
            st.success("Result: Real News")

# Footer
st.markdown(
    "<hr><p style='text-align: center; font-size: 14px;'>Made with Streamlit | Project by Astha</p>",
    unsafe_allow_html=True
)