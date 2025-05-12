
import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

models = {
    "Multinomial Naive Bayes (Accuracy: 98.6%)": pickle.load(open("model_mnb.pkl", "rb")),
    "Logistic Regression (Accuracy: 96.3%)": pickle.load(open("model_logreg.pkl", "rb")),
    "Random Forest (Accuracy: 95.2%)": pickle.load(open("model_rf.pkl", "rb")),
    "SVM (Accuracy: 94.8%)": pickle.load(open("model_svm.pkl", "rb"))
}

# Default selected model
default_model = "Multinomial Naive Bayes (Accuracy: 98.6%)"

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

# UI starts here
st.set_page_config(page_title="Spam Email Detection System", layout="centered")

st.title("📧 Spam Email Detection Using Machine Learning")
st.markdown("""
This system is designed to detect whether an email is **spam** or **not spam (ham)** using a variety of machine learning classifiers.  
It helps users filter out unwanted or malicious emails that could contain phishing links or scams.  
Such systems are essential for maintaining **email security**, reducing **clutter**, and improving **productivity**.
""")

# Input form
email_text = st.text_area("Enter the email text below:", height=200)

# Classifier selection
selected_model_label = st.selectbox("Choose a classifier:", list(models.keys()), index=list(models.keys()).index(default_model))
selected_model = models[selected_model_label]

if st.button("Analyze"):
    if email_text.strip() == "":
        st.warning("Please enter some email content.")
    else:
        cleaned_text = preprocess_text(email_text)
        vect_text = vectorizer.transform([cleaned_text])
        prediction = selected_model.predict(vect_text)[0]

        if prediction == 1:
            st.error("🚫 This email is classified as **SPAM**.")
        else:
            st.success("✅ This email is classified as **NOT SPAM**.")
