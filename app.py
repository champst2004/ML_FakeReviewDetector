import streamlit as st
from src.predict import predict

st.title("Fake Review Detector")

text = st.text_area("Enter review")

if st.button("Predict"):
    result = predict(text)
    st.write(result)