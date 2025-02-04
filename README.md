# 📊 Analyse de Sentiments des Tweets avec Dashboard Interactif  

Ce projet permet d’analyser des tweets à l’aide d’une base de données SQLite et de générer des visualisations interactives via **Gradio**. Il inclut des analyses approfondies des sentiments, des hashtags, des retweets, et des tendances temporelles.

---

## 🚀 Fonctionnalités  

✔️ **Analyse des sentiments** (positif, neutre, négatif)  
✔️ **Évolution des sentiments** dans le temps  
✔️ **Nuage de mots dynamique** selon le sentiment  
✔️ **Analyse des hashtags les plus utilisés**  
✔️ **Tendances des retweets**  
✔️ **Raisons des avis négatifs**  
✔️ **Répartition des utilisateurs par fuseau horaire**  
✔️ **Tableau interactif des tweets**  

---

## 📂 Structure du projet  

📁 projet_analyse_sentiments │── 📄 app.py # Script principal du projet │── 📄 paths.py # Fichier définissant les chemins des fichiers et de la DB │── 📂 data # Dossier pour stocker les données │ ├── database.sqlite # Base de données des tweets enrichis │── 📂 images # Stockage des visualisations générées │ ├── wordcloud.png # Nuage de mots │ ├── sentiment_trend.png # Évolution des sentiments │ ├── retweets_trend.png # Évolution des retweets │ ├── hashtags_trend.png # Tendances des hashtags │ ├── negative_reasons.png # Raisons négatives les plus fréquentes │ ├── user_timezones.png # Fuseaux horaires des utilisateurs │── 📄 requirements.txt # Liste des dépendances │── 📄 README.md # Documentation du projet

yaml
Copier
Modifier

---

## 🛠️ Installation et Utilisation  

### 1️⃣ Prérequis  

Avant de commencer, assurez-vous d’avoir :  
- **Python 3.8+** installé  
- **pip** installé  

### 2️⃣ Installation des dépendances  

Cloner le projet et installer les bibliothèques nécessaires avec :  

```bash
git clone https://github.com/Elmf95/AI_SOCIALMEDIA_ANALYSIS.git
pip install -r requirements.txt

### 3️⃣ Configuration de la base de données  

Vérifiez que le fichier **database.sqlite** est bien présent dans le dossier `data/`.  
Si vous utilisez une autre base SQLite, mettez à jour `DB_PATH` dans `paths.py` :  

```python
# Modifier le chemin de la base de données si nécessaire
DB_PATH = "data/database.sqlite"

### 4️⃣ Exécution du projet  

Lancer l’interface interactive avec la commande :  

```bash
python .\interactive_dashboard.py
