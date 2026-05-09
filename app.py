import streamlit as st
from news_verification import verify_news_online
import joblib
import re
import string
import matplotlib.pyplot as plt

# Load trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Page settings
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 AI Fake News Detection System")

st.markdown("""
This system uses **Natural Language Processing (NLP)** and **Machine Learning**
to classify news articles as **Fake** or **Real**.
""")

# User input
news_input = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

# Detect button
if st.button("Detect News"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:

        # Clean text
        cleaned_news = clean_text(news_input)

        # Transform text
        vectorized_news = vectorizer.transform([cleaned_news])

        # Prediction
        prediction = model.predict(vectorized_news)[0]
        # Verify online
        verified, results = verify_news_online(news_input)

        st.subheader("Online Verification")

        if verified:
         st.success(f"News found online in {results} articles.")
        else:
         st.error("News not found in trusted online sources.")
        # Confidence score
        probabilities = model.predict_proba(vectorized_news)[0]

        fake_confidence = probabilities[0] * 100
        real_confidence = probabilities[1] * 100

        st.subheader("Prediction Result")

        # Show result
        if prediction == 0:
            st.error("⚠️ Fake News Detected")
        else:
            st.success("✅ Real News")

        # Confidence scores
        st.write(f"### Confidence Scores")
        st.write(f"Fake News: {fake_confidence:.2f}%")
        st.write(f"Real News: {real_confidence:.2f}%")

        # Visualization
        labels = ['Fake', 'Real']
        values = [fake_confidence, real_confidence]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Confidence %")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Developed using Python, NLP, Scikit-learn & Streamlit")