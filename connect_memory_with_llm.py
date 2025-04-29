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

# --- Step 1: Get company summary
rag_context = run_hybrid_rag("Summarize the company profile.")

# --- Step 2: Format ad generation prompt
MISTRAL_PROMPT_TEMPLATE = """
🎯 You are a top-tier creative copywriter, specializing in writing engaging and persuasive product ads.

Your mission is to **generate a captivating product ad** using the given information below.
Be creative, highlight unique selling points, and make it emotionally appealing to the audience. 🌟

---

🏢 Company Context:
{context}

🛍️ Product Description:
{question}

🎨 Motif (Tone/Emotion to Use):
{motif}

---

✨ Rules:
- Start with a catchy headline
- Highlight key features naturally
- Make it feel persuasive but NOT robotic
- Use the motif/emotion strongly
- Keep it concise (max 120 words)
- Sprinkle emojis naturally
- Avoid any unrelated content

---

Now, generate the product ad below:
"""

full_prompt = MISTRAL_PROMPT_TEMPLATE.format(
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