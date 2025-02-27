import sys
import os
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import gradio as gr
from collections import Counter
from wordcloud import WordCloud

# Ajouter le chemin dynamique pour accéder à 'paths.py'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import (
    DATA_DIR,
    DB_PATH,
)  # Assure-toi que les chemins sont correctement importés depuis ton fichier paths.py


# Connexion à la base de données SQLite
def connect_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Connexion réussie à la base de données.")
        return conn
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données: {e}")
        return None


# Fonction pour récupérer les données des tweets
def fetch_data():
    conn = connect_db()
    query = "SELECT * FROM TweetsEnriched"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Ajouter un filtre global basé sur le sentiment
def filter_data_by_sentiment(df, sentiment):
    if sentiment != "Tous":
        df = df[df["airline_sentiment"] == sentiment]
    return df


# Fonction pour analyser l'évolution des sentiments dans le temps
def plot_sentiment_trend(df, sentiment, start_date=None, end_date=None):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    df["tweet_created"] = pd.to_datetime(df["tweet_created"])
    if start_date and end_date:
        df = df[(df["tweet_created"] >= start_date) & (df["tweet_created"] <= end_date)]
    trend = (
        df.groupby([df["tweet_created"].dt.date, "airline_sentiment"]).size().unstack()
    )
    trend.plot(kind="line", figsize=(10, 6))
    plt.title("évolution des sentiments au fil du temps")
    plt.xlabel("Date")
    plt.ylabel("Nombre de tweets")
    plt.tight_layout()
    plt.savefig("sentiment_trend.png")
    return "sentiment_trend.png"


# Fonction pour générer un nuage de mots par sentiment
def plot_wordcloud(df, sentiment):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    text = " ".join(df["text"].dropna())
    if not text.strip():
        return f"Aucun tweet trouv\u00e9 pour le sentiment : {sentiment}."
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    output_path = f"wordcloud_{sentiment}.png"
    wordcloud.to_file(output_path)
    return output_path


# Fonction pour analyser l'évolution des retweets dans le temps
def plot_retweets_trend(df, sentiment):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    df["tweet_created"] = pd.to_datetime(df["tweet_created"])
    trend = df.groupby(df["tweet_created"].dt.date)["retweet_count"].sum()
    trend.plot(kind="line", figsize=(10, 6))
    plt.title("évolution des retweets dans le temps")
    plt.xlabel("Date")
    plt.ylabel("Nombre de retweets")
    plt.tight_layout()
    plt.savefig("retweets_trend.png")
    return "retweets_trend.png"


# Fonction pour afficher les hashtags les plus fréquents
def plot_hashtags(df, sentiment):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    hashtags = df["hashtags"].dropna().str.split(",").explode().str.strip()
    hashtag_counts = Counter(hashtags)
    most_common_hashtags = hashtag_counts.most_common(10)
    hashtags = [hashtag for hashtag, _ in most_common_hashtags]
    counts = [count for _, count in most_common_hashtags]
    plt.figure(figsize=(10, 6))
    plt.barh(hashtags, counts)
    plt.xlabel("Fréquence")
    plt.title("Tendances des hashtags")
    plt.tight_layout()
    plt.savefig("hashtags_trend.png")
    return "hashtags_trend.png"


# Fonction pour analyser les raisons négatives les plus fréquentes
def plot_negative_reasons(df, sentiment):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    reasons = df["negativereason"].dropna()
    reason_counts = reasons.value_counts().head(10)
    plt.figure(figsize=(10, 6))
    reason_counts.plot(kind="barh")
    plt.title("Raisons négatives les plus fréquentes")
    plt.xlabel("Fréquence")
    plt.tight_layout()
    plt.savefig("negative_reasons.png")
    return "negative_reasons.png"


# Fonction pour afficher l'analyse des fuseaux horaires des utilisateurs
def plot_user_timezones(df, sentiment):
    df = filter_data_by_sentiment(df, sentiment)  # Appliquer le filtre
    timezones = df["user_timezone"].dropna().value_counts().head(10)
    plt.figure(figsize=(10, 6))
    timezones.plot(kind="barh")
    plt.title("Fuseaux horaires des utilisateurs")
    plt.xlabel("Nombre de tweets")
    plt.tight_layout()
    plt.savefig("user_timezones.png")
    return "user_timezones.png"


# Fonction pour créer un tableau interactif des tweets
def plot_interactive_table(df):
    return df


# Fonction principale du dashboard
def dashboard(sentiment):
    df = fetch_data()

    # Appliquer le filtre à chaque visualisation
    sentiment_path = plot_sentiment_trend(df, sentiment)
    wordcloud_path = plot_wordcloud(df, sentiment)
    interactive_table = plot_interactive_table(filter_data_by_sentiment(df, sentiment))
    retweets_path = plot_retweets_trend(df, sentiment)
    hashtags_path = plot_hashtags(df, sentiment)
    negative_reasons_path = plot_negative_reasons(df, sentiment)
    user_timezones_path = plot_user_timezones(df, sentiment)

    return (
        sentiment_path,
        wordcloud_path,
        interactive_table,
        retweets_path,
        hashtags_path,
        negative_reasons_path,
        user_timezones_path,
    )


# Interface utilisateur avec Gradio utilisant gr.Blocks
def create_dashboard():
    sentiment_options = ["Tous", "positive", "neutral", "negative"]

    with gr.Blocks() as interface:
        with gr.Row():
            sentiment_input = gr.Dropdown(
                sentiment_options, label="Choisir le sentiment"
            )

        with gr.Row():
            with gr.Column():
                sentiment_img = gr.Image(label="évolution des sentiments")
                wordcloud_img = gr.Image(label="Nuage de mots")
            with gr.Column():
                retweets_img = gr.Image(label="évolution des retweets")
                hashtags_img = gr.Image(label="Tendances des hashtags")
            with gr.Column():
                negative_reasons_img = gr.Image(
                    label="Raisons négatives les plus fréquentes"
                )
                user_timezones_img = gr.Image(label="Fuseaux horaires des utilisateurs")

        interactive_table = gr.DataFrame(label="Tableau interactif des tweets")

        sentiment_input.change(
            fn=dashboard,
            inputs=[sentiment_input],
            outputs=[
                sentiment_img,
                wordcloud_img,
                interactive_table,
                retweets_img,
                hashtags_img,
                negative_reasons_img,
                user_timezones_img,
            ],
        )

    return interface


# Lancer le dashboard
if __name__ == "__main__":
    create_dashboard().launch()
