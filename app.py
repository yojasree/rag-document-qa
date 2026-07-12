"""
app.py - Streamlit web interface for the RAG demo.
Run: streamlit run app.py
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from qa_chain import answer_question

st.set_page_config(page_title="Document Q&A (RAG Demo)", page_icon="📄")

st.title("Document Q&A")
st.caption("Ask questions about your document. Answers are generated using Retrieval-Augmented Generation (RAG).")

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Searching document and generating answer..."):
        answer, sources = answer_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.caption(f"Sources: page(s) {sources}")
