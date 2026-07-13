"""
ingest.py - Loads a PDF, splits it into chunks, creates embeddings, stores in ChromaDB.
Run: python src/ingest.py
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "data/research_paper.pdf"
CHROMA_DB_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_and_split_pdf(pdf_path):
    print(f"Loading PDF from {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} page(s).")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks

def build_vector_store(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Creating vector store...")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()
    print(f"Vector store saved to '{CHROMA_DB_DIR}/'.")
    return vector_store

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"No PDF found at '{PDF_PATH}'. Add one, then run this again.")
    else:
        chunks = load_and_split_pdf(PDF_PATH)
        build_vector_store(chunks)
        print("\nDone!")
