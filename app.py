import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Setup stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --- STYLING ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .stTextArea textarea {background-color: #ffffff; font-size:16px;}
        .stButton>button {background-color: #2b8a3e; color: white; font-size: 18px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📝 Customer Review Sentiment Analyzer")
st.markdown("<h4 style='color:#333333;'>Check if a review is Positive or Negative</h4>", unsafe_allow_html=True)

# --- INPUT AREA ---
review = st.text_area("✍️ Enter a product review:")

# --- PREDICTION ---
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == "positive":
            st.success("🟢 Positive Review")
        else:
            st.error("🔴 Negative Review")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Abdul Rehman 💻</p>", unsafe_allow_html=True)
