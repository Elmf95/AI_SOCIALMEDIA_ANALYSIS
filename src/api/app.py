import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ajouter le dossier racine au chemin
import sys

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import MODEL_DIR, MODEL_FILE

# Charger le modèle et le vectorizer
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Modèle et vectorizer chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle/vectorizer : {e}")
    model = None
    vectorizer = None

# Initialiser l'application FastAPI
app = FastAPI(title="API de Prédiction des Sentiments", version="1.0")


class Tweet(BaseModel):
    text: str


@app.get("/")
def read_root():
    """
    Endpoint racine pour vérifier l'état de l'API.
    """
    return {"message": "Bienvenue dans l'API de prédiction des sentiments !"}


@app.post("/predict/")
def predict_sentiment(tweet: Tweet):
    """
    Endpoint pour prédire le sentiment d'un tweet.
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Modèle non disponible.")

    # Prétraiter et vectoriser le texte
    try:
        tweet_tfidf = vectorizer.transform([tweet.text])
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur de traitement du texte : {e}"
        )

    # Faire une prédiction
    prediction = model.predict(tweet_tfidf)[0]
    proba = model.predict_proba(tweet_tfidf).max()

    return {
        "text": tweet.text,
        "prediction": int(prediction),  # Aligner avec Gradio
        "confidence": float(proba),
    }


if __name__ == "__main__":
    # Lancer le serveur avec Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
