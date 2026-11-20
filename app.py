import pandas as pd
import streamlit as st

import predict_distilbert   # DistilBERT fine-tuné (modèle retenu) : imports lourds faits à la demande

SEUIL_INCERTAIN = 0.6   # seuil défini avec P1 : sous 0,6, l'accuracy de son modèle chute nettement

MODELES = {
    "DistilBERT fine-tuné (modèle retenu)": "bert",
    "Régression logistique TF-IDF (léger, P1)": "logreg",
}
INFOS = {
    "bert": "F1 macro test : 0,9588. Transformer fine-tuné, plus lourd et plus lent.",
    "logreg": "F1 macro test : 0,9567. Modèle léger : quasi instantané, explicable mot par mot.",
}


@st.cache_resource(show_spinner="Chargement du modèle...")
def get_predictor(kind):
    """Charge le modèle une seule fois (mis en cache) et renvoie sa fonction predict."""
    if kind == "bert":
        fn = predict_distilbert.predict
    else:
        from predict import predict as fn          # modèle TF-IDF de P1
    fn("warm up: wireless usb cable")              # force le chargement du modèle maintenant
    return fn


st.set_page_config(page_title="Classification de textes e-commerce", page_icon="🛒")
st.title("🛒 Classification de descriptions de produits")
st.write("Entrez une description de produit **en anglais** : le modèle prédit sa catégorie "
         "parmi Books, Clothing & Accessories, Electronics et Household.")

# --- Choix du modèle (DistilBERT par défaut s'il est présent, sinon le modèle de P1) ---
noms = list(MODELES)
defaut = 0 if predict_distilbert.is_available() else 1
choix = st.sidebar.radio("Modèle", noms, index=defaut)
kind = MODELES[choix]
st.sidebar.caption(INFOS[kind])

tab_text, tab_file = st.tabs(["Écrire un texte", "Uploader un fichier .txt"])

text = ""
with tab_text:
    typed = st.text_area("Description du produit", height=180,
                         placeholder="e.g. Wireless Bluetooth headphones with 20h battery life")
    if typed:
        text = typed
with tab_file:
    uploaded = st.file_uploader("Fichier texte (.txt)", type=["txt"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.text_area("Contenu du fichier", text, height=150, disabled=True)

if st.button("Classifier", type="primary"):
    try:
        predict_fn = get_predictor(kind)
        result = predict_fn(text)
    except ValueError as e:                        # texte vide
        st.error(str(e))
    except Exception as e:                         # modèle absent, paquet manquant...
        st.error(f"Impossible de charger le modèle « {choix} » : {e}")
    else:
        probas = result["probas"]
        confiance = probas[result["classe"]]
        if confiance < SEUIL_INCERTAIN:
            st.warning(f"Classification incertaine : {result['classe']} ({confiance:.1%}). "
                       "Description peut-être trop courte, ambiguë ou hors des 4 catégories.")
        else:
            st.success(f"Catégorie prédite : **{result['classe']}** ({confiance:.1%})")
        df = pd.DataFrame({"Probabilité": probas}).sort_values("Probabilité", ascending=False)
        try:
            st.bar_chart(df, horizontal=True)       # barres horizontales (Streamlit récent)
        except TypeError:
            st.bar_chart(df)                        # repli pour une version plus ancienne
        st.dataframe(df.style.format("{:.1%}"))