import re
import nltk
import spacy

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download('en_core_web_sm')

# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def normalize_text(text: str) -> str:
    doc = nlp(text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)
    return normalized_text
