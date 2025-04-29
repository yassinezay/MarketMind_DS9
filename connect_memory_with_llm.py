import os
import requests
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Setup Embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Setup Mistral LLM via Ollama
def load_llm():
    return Ollama(model="mistral:7b-instruct")  # ✅ Corrected model name

# --- Prompt Template for Summarizing Company Profile
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

# --- Setup Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': get_rag_prompt()}
)

# --- Terminal Input
user_query = input("💡 Enter your product idea: ")
motif = input("🎨 Enter the desired motif/emotion (e.g., Excitement, Elegance, Trust): ")

# --- Step 1: Extract Company Context
rag_context = qa_chain.invoke({'query': 'Summarize the company profile.'})["result"]

# --- Step 2: Create Prompt for Mistral Model (via Ollama)
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

# --- Step 3: Call Mistral via Ollama (Local model)
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "mistral:7b-instruct", "prompt": full_prompt, "stream": False}  # ✅ Corrected model name
)

# --- Step 4: Show Result
result = response.json()
generated_ad = result.get("response", "").strip()

print("\n📝 Generated Ad:\n")
print(generated_ad)
