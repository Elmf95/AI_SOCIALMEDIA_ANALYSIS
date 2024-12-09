import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
import sys

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import FILTERED_DATA_FILE


# Charger les données filtrées
def load_filtered_tweets(file_path):
    try:
        data = pd.read_csv(file_path, encoding="utf-8")
        print(f"Chargement réussi de {file_path}.")
        print(f"Nombre de tweets : {data.shape[0]}")
        return data
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None


# Afficher des statistiques descriptives
def analyze_tweets(data):
    # Afficher la longueur des tweets
    data["text_length"] = data["text"].str.len()
    print("\n--- Longueur des tweets ---")
    print(data["text_length"].describe())

    # Distribution de la longueur des tweets
    plt.figure(figsize=(10, 6))
    sns.histplot(data["text_length"], bins=30, kde=True, color="blue")
    plt.title("Distribution de la longueur des tweets")
    plt.xlabel("Longueur du texte")
    plt.ylabel("Fréquence")
    plt.show()

    # Afficher les tweets les plus longs et les plus courts
    print("\n--- Tweets les plus longs ---")
    print(data.loc[data["text_length"].idxmax()]["text"])
    print("\n--- Tweets les plus courts ---")
    print(data.loc[data["text_length"].idxmin()]["text"])


# Générer un nuage de mots
def generate_wordcloud(data):
    all_text = " ".join(data["text"].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        all_text
    )

    # Afficher le nuage de mots
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nuage de mots des tweets filtrés")
    plt.show()


if __name__ == "__main__":
    # Charger les données filtrées
    filtered_data = load_filtered_tweets(FILTERED_DATA_FILE)

    if filtered_data is not None:
        # Analyser les tweets
        analyze_tweets(filtered_data)

        # Générer un nuage de mots
        generate_wordcloud(filtered_data)
    else:
        print("Aucune donnée à explorer.")
