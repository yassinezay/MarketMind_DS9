import os
import requests
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# ✅ Updated imports for HuggingFace and Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import re
from bark_audio_generator import generate_audio, save_audio_to_wav

# --- Load environment variables
load_dotenv()

# --- Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
faiss_retriever = faiss_db.as_retriever(search_kwargs={'k': 3})

# --- Web fallback (DuckDuckGo)
duckduckgo_tool = DuckDuckGoSearchRun()

def web_fallback_retriever(query: str):
    results = duckduckgo_tool.run(query)
    return [Document(page_content=results)]

# --- Hybrid retriever using new retriever method
def hybrid_retriever(query: str):
    local_docs = faiss_retriever.invoke(query)
    combined_text = " ".join([doc.page_content for doc in local_docs])
    if not local_docs or len(combined_text) < 300:
        print("🔍 Local docs insufficient. Falling back to web (DuckDuckGo)...")
        web_docs = web_fallback_retriever(query)
        return local_docs + web_docs
    return local_docs

dialect_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dialect_db = FAISS.load_local("vectorstore/dialect_faiss", dialect_embedding_model, allow_dangerous_deserialization=True)
dialect_retriever = dialect_db.as_retriever(search_kwargs={'k': 5})

# --- Load Mistral via Ollama
def load_llm():
    return OllamaLLM(model="mistral:7b-instruct")

# --- RAG prompt
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Based on the documents below, extract key insights about the company’s identity, mission, and product quality.

Documents:
{context}

Question: What is the summarized description of the company?
"""

def get_rag_prompt():
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

qa_chain = create_stuff_documents_chain(
    llm=load_llm(),
    prompt=get_rag_prompt()
)

def run_hybrid_rag(query):
    docs = hybrid_retriever(query)
    return qa_chain.invoke({"context": docs, "question": query})

# --- Input
user_query = input("💡 Enter your product idea: ")
motif = input("🎨 Enter the desired motif/emotion (e.g., Excitement, Elegance, Trust): ")
language = input("🌍 Choose ad language (English or French): ").strip().lower()

# --- Step 1: Get company summary
rag_context = run_hybrid_rag("Summarize the company profile.")
# --- Récupération des expressions en dialecte tunisien
tunisian_context_docs = dialect_retriever.invoke(user_query)
tunisian_sentences = [doc.page_content for doc in tunisian_context_docs]


# --- Step 2: Format ad generation prompt
MISTRAL_PROMPT_EN = """
🚀 You are a world-class creative copywriter trusted by top global brands to craft unforgettable, high-converting product advertisements.

Your job is to write a magnetic product ad that emotionally connects with the audience, showcases the product's unique value, and cleverly integrates the Tunisian cultural flair using the provided expressions. 🎯

---

💬 Use at least one of the following authentic Tunisian expressions in each sentence to make it vibrant and locally resonant:
{tunisian_phrases}

🏢 **Company Essence:**
{context}

🛍️ **Product Description (Idea):**
{question}

🎨 **Creative Motif / Emotion to Embody:**
{motif}

---

🛠️ **Creative Rules:**
- Start with a **bold, catchy headline** that grabs attention immediately
- Use storytelling to evoke **emotion** and build desire
- Blend **benefits** and **feelings**, not just features
- Use **modern, informal tone** that feels human — not robotic
- Inject **natural emojis** that enhance, not distract
- Stay under **120 words**
- Finish with **3–5 punchy hashtags** (e.g., #AuthenticVibes, #TunisianPride)
- Absolutely avoid repetition, clichés, or boring wording

---

✍️ Now, craft a powerful ad below. Make it scroll-stopping, fun, and unforgettable:
"""

MISTRAL_PROMPT_FR = """
🚀 Tu es un·e copywriter créatif·ve de renommée mondiale, choisi·e par les plus grandes marques pour créer des publicités produits inoubliables et ultra-convertissantes.

Ta mission : rédiger une publicité percutante qui touche le cœur du public, valorise l’unicité du produit, et intègre avec créativité des expressions tunisiennes authentiques. 🎯

---

💬 Utilise au moins une des expressions tunisiennes suivantes dans chaque phrase pour donner du caractère et une touche locale :
{tunisian_phrases}

🏢 **Essence de l'entreprise :**
{context}

🛍️ **Description du produit :**
{question}

🎨 **Motif / Émotion à transmettre :**
{motif}

---

🛠️ **Consignes créatives :**
- Commence par un **titre accrocheur** et impactant
- Raconte une histoire qui suscite **l’émotion** et donne envie
- Mets en valeur les **bénéfices**, pas seulement les caractéristiques
- Utilise un **ton naturel et moderne** – pas de robot !
- Intègre des **emojis pertinents** de manière fluide
- Ne dépasse pas **120 mots**
- Termine par **3 à 5 hashtags puissants** (ex. : #FiertéTunisienne, #StyleMaghreb)
- Évite absolument la redondance, les clichés ou les tournures fades

---

✍️ Maintenant, écris une pub puissante. Elle doit captiver, faire sourire et marquer les esprits :
"""



if language == "french":
    selected_prompt = MISTRAL_PROMPT_FR
else:
    selected_prompt = MISTRAL_PROMPT_EN

full_prompt = selected_prompt.format(
    tunisian_phrases="\n".join(tunisian_sentences),
    context=rag_context,
    question=user_query,
    motif=motif
)

# --- Step 3: Send to LLaMA 3 (Ollama endpoint)
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": full_prompt, "stream": False}
)

generated_ad = response.json().get("response", "").strip()

print("\n📝 Generated Ad:\n")
if generated_ad:
    print(generated_ad)
else:
    print("⚠️ No content generated. Please check the prompt or if the model is active.")

def extract_headline(ad_text):
    """Extracts the headline following the '**Headline:**' label."""
    match = re.search(r"\*\*Headline:\*\*\s*(.+)", ad_text)
    if match:
        # Optionally clean up markdown or emojis if needed
        return match.group(1).strip()
    return None

headline = extract_headline(generated_ad)
if headline:
    print(f"\n🗣️ Generating audio for headline: \"{headline}\"\n")
    audio_array, sample_rate = generate_audio(headline)
    save_audio_to_wav(audio_array, sample_rate, filename="ad_headline_audio.wav")
else:
    print("⚠️ No headline found to generate audio.")