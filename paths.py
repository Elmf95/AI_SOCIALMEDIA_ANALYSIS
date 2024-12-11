import os

# Chemin de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le dossier des données
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
FILTERED_DATA_FILE = os.path.join(DATA_DIR, "filtered_tweets.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_tweets_train.csv")
DB_PATH = os.path.join(DATA_DIR, "database.sqlite")
CSV_PATH = os.path.join(DATA_DIR, "Tweets.csv")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "cleaned_sqlite.csv")

# Chemin vers le dossier des modèles
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = "sentiment_model.joblib"

# Chemin vers le dossier des vecteurs
VECTOR_DIR = os.path.join(BASE_DIR, "vectors")
VECTOR_FILE = "tfidf_vectors.joblib"
