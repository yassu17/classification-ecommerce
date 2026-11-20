"""Contrat d'interface du binôme : predict(text) -> {"classe": str, "probas": {classe: proba}}."""
from pathlib import Path

import joblib
import numpy as np

import preprocessing  # noqa: F401  (nécessaire pour charger le Pipeline sauvegardé)

CLASSES = ["Books", "Clothing & Accessories", "Electronics", "Household"]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_classic.joblib"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        assert list(_model.classes_) == list(range(len(CLASSES))), "Ordre des classes inattendu"
    return _model


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte est vide.")
    proba = _get_model().predict_proba([text])[0]
    return {"classe": CLASSES[int(np.argmax(proba))],
            "probas": {c: float(p) for c, p in zip(CLASSES, proba)}}
