import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ajout du chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH


# Charger les données de SQLite
def load_merged_data(path, table_name="Tweets"):
    print("Chargement des données fusionnées depuis SQLite...")
    conn = sqlite3.connect(path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Analyser la distribution des sentiments
def analyze_sentiment_distribution(df, db_path):
    print("Analyse de la distribution des sentiments...")
    plt.figure(figsize=(8, 6))
    sns.countplot(
        data=df,
        x="airline_sentiment",
        order=df["airline_sentiment"].value_counts().index,
    )
    plt.title("Distribution des sentiments")
    plt.xlabel("Sentiment")
    plt.ylabel("Nombre de tweets")
    plt.tight_layout()

    # Définir le chemin pour sauvegarder le graphique
    output_dir = os.path.join(os.path.dirname(db_path), "plots")
    os.makedirs(output_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas
    plot_path = os.path.join(output_dir, "sentiment_distribution.png")

    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans : {plot_path}")
    plt.show()


# Identifier les données manquantes
def analyze_missing_data(df):
    print("Analyse des données manquantes...")
    missing_data = df.isnull().sum()
    missing_report = missing_data[missing_data > 0].sort_values(ascending=False)
    print("Colonnes avec des valeurs manquantes :")
    print(missing_report)
    return missing_report


# Sauvegarder le rapport des données manquantes
def save_missing_data_report(missing_report, db_path):
    # Définir le répertoire de destination pour le rapport
    output_dir = os.path.join(os.path.dirname(db_path), "reports")
    os.makedirs(output_dir, exist_ok=True)  # Créer le répertoire s'il n'existe pas

    # Chemin complet pour le rapport
    report_path = os.path.join(output_dir, "missing_data_report.csv")

    # Sauvegarder le rapport
    missing_report.to_csv(report_path, header=True)
    print(f"Rapport des données manquantes sauvegardé dans : {report_path}")


if __name__ == "__main__":
    # Charger les données fusionnées
    df = load_merged_data(DB_PATH)

    # Aperçu des données
    print("Aperçu des données :\n", df.head())
    print(f"Nombre total de lignes : {len(df)}")

    # Analyse des sentiments
    analyze_sentiment_distribution(df, DB_PATH)

    # Analyse des données manquantes
    missing_report = analyze_missing_data(df)

    # Sauvegarder le rapport des données manquantes
    save_missing_data_report(missing_report, DB_PATH)
