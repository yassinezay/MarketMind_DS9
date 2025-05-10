import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load CSV with semicolon separator
df = pd.read_csv("dictionnaire_tunisien.csv", sep=";")

# Create documents from Tunisian dialect phrases
docs = [Document(page_content=row["tunisien"]) for _, row in df.iterrows()]

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create FAISS vector store
db = FAISS.from_documents(docs, embedding_model)

# Save FAISS index
db.save_local("vectorstore/dialect_faiss")