import pandas as pd
from sentence_transformers import SentenceTransformer, util
import io
import base64
import matplotlib.pyplot as plt 
import seaborn as sns



def load_model(model_path: str):
    """Charger le modèle SentenceTransformer depuis un dossier local."""
    model = SentenceTransformer(model_path)
    return model


def clean_text(text):
    if isinstance(text, str):
        return text.encode('latin1').decode('utf-8', errors='ignore')
    return text


def load_and_clean_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding='latin1', sep=';')
    for col in df.columns:
        df[col] = df[col].apply(clean_text)
    return df


def compute_embeddings(model, texts):
    return model.encode(texts)


def recommander_strategie(commentaires, model, df_strat, strategy_embeddings):
    recommandations = []

    if isinstance(commentaires, str):
        commentaires = [commentaires]

    comment_embeddings = compute_embeddings(model, commentaires)

    for i, comment in enumerate(commentaires):
        similarities = util.cos_sim(comment_embeddings[i], strategy_embeddings)[0]
        best_idx = similarities.argmax().item()
        strat = df_strat.iloc[best_idx]

        recommandations.append({
            "Commentaire": comment,
            "Stratégie recommandée": strat["Stratégie"],
            "Objectif": strat["Objectif"],
            "Description": strat["Description"]
        })

    return recommandations


import io
import base64

def plot_recommandations(recommandations_df):
    strategie_counts = recommandations_df['Stratégie recommandée'].value_counts().reset_index()
    strategie_counts.columns = ['Stratégie', 'Nombre']

    plt.figure(figsize=(10, 6))
    sns.barplot(data=strategie_counts, x='Nombre', y='Stratégie', palette='viridis')
    plt.title("Stratégies marketing recommandées en fonction des commentaires")
    plt.xlabel("Nombre de fois recommandée")
    plt.ylabel("Stratégie")
    plt.tight_layout()

    # Sauvegarde du graphique dans un buffer mémoire
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Encodage en base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64



def export_recommandations_csv(recommandations_df, path):
    recommandations_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ Fichier exporté : {path}")
