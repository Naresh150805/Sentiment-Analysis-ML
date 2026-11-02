import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# Load saved model
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    cleaned = clean_text(review)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == "positive":
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")