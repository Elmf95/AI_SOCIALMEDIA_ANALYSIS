import requests
import gradio as gr

# URL de l'API Flask
API_URL = "http://127.0.0.1:8000/predict/"  # Port corrigé pour FastAPI


def analyze_sentiment(text):
    """
    Fonction pour interagir avec l'API FastAPI.
    Envoie un texte à l'API et retourne la prédiction et la confiance.
    """
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            result = response.json()
            prediction = "Positive" if result["prediction"] == 1 else "Negative"
            confidence = result["confidence"]
            return f"Sentiment: {prediction} (Confidence: {confidence:.2f})"
        else:
            return f"Erreur de l'API : {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Erreur lors de la connexion à l'API : {e}"


# Interface Gradio
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Analyze the sentiment of a given text using our trained model. (rentrez votre prompt en anglais svp, le modèle a été entraîné uniquement sur des datasets anglais)",
)

if __name__ == "__main__":
    interface.launch()
