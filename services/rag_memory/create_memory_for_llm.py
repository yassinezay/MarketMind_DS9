# services/rag_memory/connect_memory_for_llm.py

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os

DB_FAISS_PATH = os.path.join("templatemo_562_space_dynamic", "vectorstore", "db_faiss")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create the FAISS DB
def load_faiss_db():
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return None

faiss_db = load_faiss_db()
faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 3}) if faiss_db else None

def hybrid_retriever(query: str, enable_fallback: bool = True):
    docs = faiss_retriever.invoke(query) if faiss_retriever else []

    if enable_fallback and (not docs or sum(len(doc.page_content) for doc in docs) < 300):
        try:
            fallback = DuckDuckGoSearchRun().run(query)
            docs.append(Document(page_content=fallback))
        except Exception as e:
            print("⚠️ DuckDuckGo fallback failed:", str(e))
            docs.append(Document(page_content=""))
    
    return docs

def run_hybrid_rag(query, enable_fallback=True):
    llm = OllamaLLM(model="mistral:7b-instruct")
    prompt = PromptTemplate(
        template="""
You are a helpful assistant. Based on the documents below, extract key insights about the company’s identity, mission, and product quality.

Documents:
{context}

Question: What is the summarized description of the company?
""",
        input_variables=["context", "question"]
    )
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    docs = hybrid_retriever(query, enable_fallback=enable_fallback)
    return chain.invoke({"context": docs, "question": query})

# NEW: Upload and embed PDF content
def store_pdf_to_faiss(pdf_path: str):
    # Define where to save the FAISS DB
    save_dir = os.path.join("templatemo_562_space_dynamic", "vectorstore", "db_faiss")
    os.makedirs(save_dir, exist_ok=True)

    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create FAISS DB and save it
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(save_dir)

    return len(chunks)

