# installing all the packages used at the end
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps=PorterStemmer()


# a function used to describe and make the site
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return "".join(y)
# pickle used to convert into nyte stream

tfidf=pickle.loads(open("vectorizer.pk1","rb"))
model=pickle.load(open("model.pkl",'rb'))

st.title("Email\SMS Spam Classifier")

input_sms=st.text_input("enter the message")

#preprocessing

transformed_sms()==transform_text(input_sms)

#Vectorize

vector_input=tfidf.transform([transformed_sms])

#Prediction

result=model.predict(vector_input)[0]


#Display

if result == 1:
    st.header("Spam")
else:
    st.header("Not Spam")












