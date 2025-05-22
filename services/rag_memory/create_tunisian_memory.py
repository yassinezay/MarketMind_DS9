# services/rag_memory/create_tunisian_memory.py

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the CSV of Tunisian expressions
def load_tunisian_phrases(csv_path: str = "dictionnaire_tunisien.csv"):
    df = pd.read_csv(csv_path, sep=";")
    docs = [Document(page_content=row["tunisien"]) for _, row in df.iterrows()]
    return docs

# Build and save FAISS index
def create_and_save_dialect_index():
    docs = load_tunisian_phrases()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local("vectorstore/dialect_faiss")

# Load dialect retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dialect_db = FAISS.load_local("vectorstore/dialect_faiss", embedding_model, allow_dangerous_deserialization=True)
dialect_retriever = dialect_db.as_retriever(search_kwargs={"k": 5})