"""
qa_chain.py - Combines retrieval with an LLM to generate grounded answers.
Needs an OPENAI_API_KEY in a .env file in your project root.
Run: python src/qa_chain.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve_relevant_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_question(question):
    chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    sources = [chunk.metadata.get("page", "unknown") for chunk in chunks]
    return answer, sources

if __name__ == "__main__":
    question = input("Ask a question about your document: ")
    answer, sources = answer_question(question)
    print("\nAnswer:")
    print(answer)
    print(f"\nSources: page(s) {sources}")
