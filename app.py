import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

st.title("📝 Customer Review Sentiment Analyzer")
review = st.text_area("Enter a product review:")

if st.button("Analyze"):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    if prediction == "positive":
        st.success("🟢 Positive Review")
    else:
        st.error("🔴 Negative Review")
