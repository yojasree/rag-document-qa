"""
retrieve.py - Searches the vector database for chunks relevant to a question.
Run: python src/retrieve.py
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DB_DIR = "chroma_db"
TOP_K = 4

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vector_store

def retrieve_relevant_chunks(question, vector_store=None):
    if vector_store is None:
        vector_store = load_vector_store()
    results = vector_store.similarity_search(question, k=TOP_K)
    return results

if __name__ == "__main__":
    test_question = "What is this document about?"
    print(f"Searching for chunks relevant to: '{test_question}'\n")
    chunks = retrieve_relevant_chunks(test_question)
    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ---")
        print(chunk.page_content[:300], "...\n")
