import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import CSV_PATH


def load_tweets(csv_path):
    """
    Charger et explorer le fichier CSV contenant les tweets.
    """
    try:
        tweets = pd.read_csv(csv_path)
        print("Aperçu des données :\n", tweets.head())
        print("Colonnes disponibles :\n", tweets.columns)
        print(f"Nombre de tweets chargés : {tweets.shape[0]}")
        return tweets
    except Exception as e:
        print(f"Erreur lors du chargement des tweets : {e}")
        return None


if __name__ == "__main__":
    tweets_df = load_tweets(CSV_PATH)
