import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):
        if text is None or text != text:
            return ""
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)