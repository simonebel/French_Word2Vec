import unicodedata
from string import digits, punctuation

import spacy
from nltk.corpus import stopwords

ADDITIONAL_STOP_WORDS = ["les", "l'", "d'", "lors", "moyen", "entre"]
stop_words = stopwords.words("french")
stop_words.extend(ADDITIONAL_STOP_WORDS)
model = spacy.load("fr_core_news_sm")

NLP_DISABLE = [
    "tagger",
    "parser",
    "ner",
    "entity_linker",
    "entity_ruler",
    "textcat",
    "textcat_multilabel",
    "morphologizer",
    "attribute_ruler",
    "senter",
    "sentencizer",
    "tok2vec",
    "transformer",
]


def clean_accents(text):
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()


def clean_token(token):
    cleaned_token = token.lemma_
    cleaned_token = cleaned_token.lower()
    if cleaned_token in stop_words:
        return ""

    cleaned_token = clean_accents(cleaned_token)

    cleaned_token = cleaned_token.translate(str.maketrans("", "", digits))

    cleaned_token = (
        cleaned_token.translate(str.maketrans(punctuation, " " * len(punctuation)))
        .replace(" " * 4, " ")
        .replace(" " * 3, " ")
        .replace(" " * 2, " ")
        .strip()
    )

    return cleaned_token


def tokenize(corpus):
    tokenized_corpus = []
    for doc in model.pipe(corpus, disable=NLP_DISABLE):
        tokenized_corpus.append(
            [clean_token(token) for token in doc if clean_token(token)]
        )

    return tokenized_corpus
