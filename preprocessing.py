"""Pré-traitement partagé entre le notebook et la webapp : même fonction à l'entraînement et en production."""
import html
import re
import warnings
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# En plus des stopwords anglais : jargon commercial présent dans toutes les catégories
# (il ne dit rien sur le TYPE de produit) et résidus du web.
EXTRA_STOPWORDS = {"buy", "online", "india", "price", "prices", "best", "low", "free", "shipping",
                   "delivery", "cash", "amazon", "flipkart", "http", "https", "www", "com", "nbsp", "amp"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    for res in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    _LEMMATIZER = WordNetLemmatizer()
    _LEMMATIZER.lemmatize("tests")        # force le chargement de WordNet
except Exception:                          # nltk absent : on prévient bruyamment
    _LEMMATIZER = None
    warnings.warn("nltk/WordNet indisponible : lemmatisation désactivée. "
                  "Installez nltk pour reproduire exactement les résultats du notebook.")


@lru_cache(maxsize=300_000)
def _lemma(word):
    return _LEMMATIZER.lemmatize(word) if _LEMMATIZER else word


def clean_text(text, lemmatize=True):
    """Nettoie une description brute et renvoie une chaîne de mots séparés par des espaces."""
    text = html.unescape(str(text))                    # &amp; -> &
    text = re.sub(r"<[^>]+>", " ", text)               # balises HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # URL
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)              # on ne garde que les lettres
    tokens = [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]
    if lemmatize:
        tokens = [_lemma(w) for w in tokens]
    return " ".join(tokens)


def clean_batch(texts, lemmatize=True):
    """Version 'liste', utilisée comme première étape du Pipeline scikit-learn."""
    return [clean_text(t, lemmatize) for t in texts]