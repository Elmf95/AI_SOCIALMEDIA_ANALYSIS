import os

# Chemin de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le dossier des données
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
FILTERED_DATA_FILE = os.path.join(DATA_DIR, "filtered_tweets.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_tweets_train.csv")
# Chemin vers le dossier des modèles
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = "sentiment_model.joblib"


# Créer les répertoires si nécessaire
os.makedirs(DATA_DIR, exist_ok=True)
import sys
import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Ajouter le dossier racine au PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

# Débogage pour vérifier les chemins
print("Python PATH:", sys.path)

from paths import PROCESSED_DATA_FILE  # Assurez-vous que ce fichier existe

# Télécharger les ressources nécessaires
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Nettoyage avancé des tweets
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    words = word_tokenize(text)
    words = [
        word
        for word in words
        if word not in stopwords.words("english") and word.isalpha()
    ]
    return " ".join(words)


# Charger les données depuis le chemin dynamique
def load_data():
    dataset = pd.read_csv(PROCESSED_DATA_FILE, encoding="utf-8")
    return dataset


# Charger et préparer les données
def prepare_data(dataset):
    dataset["clean_text"] = dataset["text"].apply(clean_tweet)
    dataset = dataset[["clean_text", "target"]]
    dataset["target"] = dataset["target"].apply(lambda x: 1 if x == 4 else 0)
    return dataset


# Pipeline optimisé pour l'entraînement du modèle
def train_model(dataset):
    X = dataset["clean_text"]
    y = dataset["target"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = ImbPipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))
    return pipeline


if __name__ == "__main__":
    dataset = load_data()
    prepared_data = prepare_data(dataset)
    model = train_model(prepared_data)
