import pickle
from src.preprocessing import preprocess

model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

def predict(text):
    cleaned = preprocess(text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)

    return "Genuine (OR)" if pred[0] == 1 else "Fake (CG)"