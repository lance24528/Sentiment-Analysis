import numpy as np
import pandas as pd
import re
import nltk
import spacy

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob, Word

# Load spaCy's language model (e.g., English)
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

dataset = pd.read_csv(
    "C:/Users/Gicano Brothers/Documents/POP Repositories/Sentiment-Analysis/data/raw/IMDB Dataset.csv"
)


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Removes HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keeping only letters
    text = text.lower()  # Converts all text to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Removes unnecessary spaces
    return text


dataset["cleaned_review"] = dataset["review"].apply(clean_text)

(dataset.head())  # Checking the result of preprocessing done in this cell

# Wrote this code to check if the NLTK punkt tokenizer is available
# Since the cell below wasnt running because in kaggle importing "punkt" was enough
# I had to download "punkt_tab" instead
try:
    word_tokenize("This is a test sentence.")
    print("NLTK punkt tokenizer is available!")
except LookupError as e:
    print(e)

    stop_words = set(stopwords.words("english"))


# Breaking down all the words into individual strings from cleaned_review column
# And removing irrelevant words like (of, and, the, is, etc.)
def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


dataset["tokenized_review"] = dataset["cleaned_review"].apply(tokenize_text)

(dataset.head())  # Checking the result of preprocessing done in this cell
# Load SpaCy's English model (disable unnecessary components for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# Function to lemmatize tokens in batches
def lemmatize_in_batches(tokenized_reviews):
    # Join tokenized reviews into strings for batch processing
    reviews_as_strings = [" ".join(tokens) for tokens in tokenized_reviews]
    # Process in batches using nlp.pipe for efficiency
    docs = nlp.pipe(reviews_as_strings, batch_size=1000)
    # Extract lemmatized tokens
    lemmatized_reviews = [[token.lemma_ for token in doc] for doc in docs]
    return lemmatized_reviews


# Apply lemmatization in batches to the tokenized_review column
dataset["lemmatized_review"] = lemmatize_in_batches(dataset["tokenized_review"])
(dataset.head())  # Checking the result of preprocessing done in this cell
