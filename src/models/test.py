import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH

# Remplacez par le bon chemin


def check_columns_in_table(db_path, table_name):
    try:
        # Connexion à la base de données SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Exécuter la commande PRAGMA pour obtenir les colonnes de la table
        cursor.execute(f"PRAGMA table_info({table_name});")

        # Récupérer les résultats (informations sur les colonnes)
        columns = cursor.fetchall()

        if columns:
            print(f"Colonnes de la table '{table_name}' :")
            for column in columns:
                # Le premier élément de chaque tuple est le nom de la colonne
                print(f"- {column[1]}")  # column[1] est le nom de la colonne
        else:
            print(f"Aucune colonne trouvée pour la table '{table_name}'.")

        # Fermer la connexion
        conn.close()
    except sqlite3.Error as e:
        print(f"Erreur SQLite : {e}")


# Appeler la fonction pour vérifier les colonnes de la table 'tweets'
check_columns_in_table(DB_PATH, "tweets")
