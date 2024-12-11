import os
import sqlite3
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import sys

# Ajout du chemin dynamique
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import DB_PATH


# Charger les données enrichies depuis SQLite
def load_enriched_data(path, table_name="TweetsEnriched"):
    conn = sqlite3.connect(path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Visualisation : évolution des sentiments au fil du temps
def plot_sentiment_trend(df, start_date=None, end_date=None):
    df["tweet_created"] = pd.to_datetime(df["tweet_created"])
    if start_date and end_date:
        df = df[(df["tweet_created"] >= start_date) & (df["tweet_created"] <= end_date)]
    trend = (
        df.groupby([df["tweet_created"].dt.date, "airline_sentiment"]).size().unstack()
    )
    trend.plot(kind="line", figsize=(10, 6))
    plt.title("Évolution des sentiments au fil du temps")
    plt.xlabel("Date")
    plt.ylabel("Nombre de tweets")
    plt.tight_layout()
    plt.savefig("trend.png")
    return "trend.png"


# Visualisation : distribution des mots-clés
def plot_keywords_distribution(df, sentiment_filter=None):
    if sentiment_filter:
        # Mapper les sentiments en français aux valeurs de la base de données
        sentiment_mapping = {
            "positif": "positive",
            "neutre": "neutral",
            "negatif": "negative",
        }
        sentiment_db = sentiment_mapping.get(sentiment_filter)
        if sentiment_db:  # Si le mapping existe
            df = df[df["airline_sentiment"] == sentiment_db]
    keywords = df["text"].str.extractall(r"#(\w+)")[0].value_counts().head(20)
    keywords.plot(kind="bar", figsize=(10, 6))
    plt.title("Distribution des mots-clés")
    plt.xlabel("Mots-clés")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig("keywords.png")
    return "keywords.png"


# Fonction principale pour l'interface
def dashboard(start_date, end_date, sentiment):
    df = load_enriched_data(DB_PATH)
    trend_path = plot_sentiment_trend(df, start_date, end_date)
    keywords_path = plot_keywords_distribution(df, sentiment)
    return trend_path, keywords_path


# Interface Gradio
def main():
    interface = gr.Interface(
        fn=dashboard,
        inputs=[
            gr.Textbox(
                placeholder="YYYY-MM-DD",
                label="Date de début (optionnel, mais à partir du 2015-02-16 minimum)",
            ),
            gr.Textbox(
                placeholder="YYYY-MM-DD",
                label="Date de fin (optionnel, mais à partir du 2015-02-24 maximum)",
            ),
            gr.Dropdown(
                ["positif", "neutre", "negatif", None],
                label="Filtrer par sentiment (optionnel)",
                value=None,
            ),
        ],
        outputs=[
            gr.Image(type="filepath", label="Évolution des sentiments"),
            gr.Image(type="filepath", label="Distribution des mots-clés"),
        ],
        title="Tableau de bord interactif",
        description="Visualisation des sentiments et des mots-clés des tweets enrichis.",
    )
    interface.launch()


if __name__ == "__main__":
    main()
