# services/text_generation/generate_ad.py

import re
import requests
from services.rag_memory.create_memory_for_llm import run_hybrid_rag
from services.rag_memory.create_tunisian_memory import dialect_retriever

# --- Prompt templates ---
MISTRAL_PROMPT_TEMPLATE_EN = """
🚀 You are a world-class creative copywriter trusted by top global brands to craft unforgettable, high-converting product advertisements.

You're crafting an ad for **{platform}**, so keep the tone aligned with that audience.

Your job is to write a magnetic product ad that emotionally connects with the audience, showcases the product's unique value, and integrates **multiple** authentic Tunisian expressions in each sentence. 🎯

💬 Here are some Tunisian phrases to use freely and repeatedly:
{tunisian_phrases}

🏢 **Company Essence:**
{context}

🛍️ **Product Description (Idea):**
{question}

🎨 **Creative Motif / Emotion to Embody:**
{motif}

🛠️ **Creative Rules:**
- Start with a bold, catchy headline
- Use storytelling to evoke emotion and build desire
- Blend benefits and feelings, not just features
- Use a modern, informal tone that feels human — not robotic
- Add natural emojis that enhance, not distract
- Stay under 120 words
- Finish with 3–5 hashtags (no label or prefix)
- Avoid repetition, clichés, or boring wording

✍️ Now, craft the ad (starting directly with the headline, no explanation):
"""

MISTRAL_PROMPT_TEMPLATE_FR = """
🚀 Tu es un·e copywriter de renommée mondiale, choisi·e par les plus grandes marques pour créer des publicités produits inoubliables et ultra-convertissantes.

Tu vas rédiger une publicité destinée à **{platform}**, donc adapte le ton à ce public.

Ta mission : écrire une publicité percutante qui touche émotionnellement le public, valorise l’unicité du produit, et intègre **plusieurs** expressions tunisiennes authentiques dans chaque phrase. 🎯

💬 Voici des expressions tunisiennes à utiliser librement et de manière répétée :
{tunisian_phrases}

🏢 **Essence de l'entreprise :**
{context}

🛍️ **Description du produit (Idée) :**
{question}

🎨 **Motif créatif / Émotion à transmettre :**
{motif}

🛠️ **Règles créatives :**
- Commencer par un titre audacieux et accrocheur
- Raconter une histoire qui suscite l’émotion et crée le désir
- Mélanger les bénéfices et les émotions, pas seulement les caractéristiques
- Utiliser un ton moderne, naturel, humain – surtout pas robotique
- Ajouter des emojis naturels qui enrichissent sans distraire
- Ne pas dépasser 120 mots
- Terminer par 3 à 5 hashtags (sans étiquette ni préfixe)
- Éviter absolument les répétitions, les clichés ou les tournures fades

✍️ Rédige maintenant la publicité (commence directement par le titre, sans explication) :
"""

# --- Prompt formatting function ---
def format_prompt(language, context, question, motif, tunisian_phrases, platform):
    base_prompt = MISTRAL_PROMPT_TEMPLATE_FR if language == "french" else MISTRAL_PROMPT_TEMPLATE_EN
    return base_prompt.format(
        tunisian_phrases="\n".join(tunisian_phrases),
        context=context,
        question=question,
        motif=motif,
        platform=platform
    )


# --- Extract the headline from the generated ad ---
def extract_headline(ad_text: str):
    match = re.search(r"\*\*Headline:\*\*\s*(.+)", ad_text)
    return match.group(1).strip() if match else None

def clean_generated_text(text: str) -> str:
    text = re.sub(r"\*\*Headline:\*\*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*Body:\*\*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Headline\s*:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Body\s*:", "", text, flags=re.IGNORECASE)
    return text.strip()

# --- Main pipeline ---
def generate_ad_pipeline(query: str, motif: str, language: str, platform: str):
    rag_context = run_hybrid_rag("Summarize the company profile.", enable_fallback=False)
    dialect_docs = dialect_retriever.invoke(query)
    tunisian_phrases = [doc.page_content for doc in dialect_docs]

    full_prompt = format_prompt(language, rag_context, query, motif, tunisian_phrases, platform)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": full_prompt, "stream": False}
    )

    ad_text = response.json().get("response", "")
    ad_text_cleaned = clean_generated_text(ad_text)
    headline = extract_headline(ad_text_cleaned)
    return ad_text_cleaned, headline
