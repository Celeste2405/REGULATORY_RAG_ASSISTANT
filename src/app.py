import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag_chain import ask

load_dotenv()

# --- UI ---
st.set_page_config(page_title="Regulatory RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Regulatory RAG Assistant")
st.caption("Bâle III/IV · IFRS9 · Stress-test · ICAAP")

question = st.text_input("Votre question réglementaire",
                         placeholder="Ex : Quelle est la définition du défaut selon Bâle ?")

if question:
    with st.spinner("Recherche en cours..."):
        result = ask(question)

    st.markdown("### Réponse")
    st.write(result["answer"])

    st.markdown("### Sources")
    for s in result["sources"]:
        title = s["title"][:80] or "—"
        ref   = s["reference"] or "—"
        url   = s.get("url", "")
        if url and url != "to_verify":
            st.markdown(f"- [{title}]({url}) — `{ref}`")
        else:
            st.markdown(f"- {title} — `{ref}`")