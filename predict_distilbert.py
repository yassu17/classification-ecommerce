"""Prédiction avec le DistilBERT fine-tuné, même contrat que predict.py de P1 :
predict(text) -> {"classe": str, "probas": {classe: proba}}

Où le modèle est-il cherché ? Dans cet ordre :
1. variable d'environnement DISTILBERT_PATH (dossier ou identifiant Hugging Face Hub) ;
2. dossier local `distilbert_ecom/` à côté de ce fichier (usage sur votre ordinateur) ;
3. dépôt de modèle sur le Hub (HUB_ID), téléchargé au premier appel (usage en ligne,
   car le modèle de 268 Mo est trop gros pour GitHub).
Les imports lourds (torch, transformers) sont faits à la demande : l'app démarre même sans eux.
"""
import importlib.util
import os
from pathlib import Path

CLASSES = ["Books", "Clothing & Accessories", "Electronics", "Household"]   # ordre des labels 0..3 à l'entraînement
MAX_LEN = 256                                                                # identique à l'entraînement
HUB_ID = "myassine23/distilbert-ecommerce"                                   # dépôt de modèle public sur Hugging Face
LOCAL_DIR = Path(__file__).resolve().parent / "distilbert_ecom"

_tokenizer = _model = _device = None


def model_source():
    """Dossier ou identifiant Hub utilisé pour charger le modèle."""
    return os.environ.get("DISTILBERT_PATH") or (str(LOCAL_DIR) if LOCAL_DIR.exists() else HUB_ID)


def is_available():
    """True si torch et transformers sont installés (sans eux, DistilBERT ne peut pas tourner)."""
    return all(importlib.util.find_spec(p) is not None for p in ("torch", "transformers"))


def _load():
    global _tokenizer, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        src = model_source()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(src)
        _model = AutoModelForSequenceClassification.from_pretrained(src).to(_device).eval()
        assert _model.config.num_labels == len(CLASSES), "Nombre de classes inattendu"
    return _tokenizer, _model, _device


def predict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte est vide.")
    import torch
    tokenizer, model, device = _load()
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        proba = torch.softmax(model(**enc).logits.float(), dim=-1)[0].cpu().numpy()
    return {"classe": CLASSES[int(proba.argmax())],
            "probas": {c: float(p) for c, p in zip(CLASSES, proba)}}


if __name__ == "__main__":
    import sys
    print("Source du modèle :", model_source())
    print(predict(" ".join(sys.argv[1:]) or "USB charging cable for smartphone, 1 metre"))