import streamlit as st
import pickle
from preprocessing import preprocess

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("Fake Product Review Detection")

user_input = st.text_area("Enter Review Text")

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Genuine Review")
    else:
        st.error("Fake Review")