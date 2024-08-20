# importing all the necessary libraries
import streamlit as st
import pickle 
import string
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
 
ps = PorterStemmer() 

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():# is alpha numaric
            y.append(i)#add function at the end

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)# pickel module

with open("vectorizer.pkl", "rb") as vec_file:
    tfidf = pickle.load(vec_file)

# Streamlit app
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)

    # Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])

    # Make prediction
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
