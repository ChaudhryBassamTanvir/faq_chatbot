import nltk
import re

nltk.download('punkt')

def preprocess_text(text: str) -> str:
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)
