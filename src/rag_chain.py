import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


#  ChromaDB : Réinitialiser le client et la collection
client = chromadb.PersistentClient(path="data/chroma_db")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection("regulatory_docs", embedding_function=embedding_fn)

# Retriever : Recherche les chunks pertinents dans ChromaDB
def retrieve(question: str, n: int = 5):
    res = collection.query(query_texts=[question], n_results=n,
                           include=["documents", "metadatas"])
    return res["documents"][0], res["metadatas"][0]

# LLM: Initialiser le modèle de langage
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0)

# Prompt : Définir le prompt pour le modèle de langage
prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en réglementation bancaire (Bâle, IFRS9, stress-test).
Réponds uniquement à partir des extraits ci-dessous.
Si l'information est absente, dis-le clairement.
Cite toujours la source entre crochets [Titre – Référence].

Contexte :
{context}"""),
    ("human", "{question}")
])

# Fonction principale 
def ask(question: str, n: int = 5) -> dict:
    docs, metas = retrieve(question, n)
    context = "\n\n---\n\n".join(
        f"[{m.get('title','')[:60]} – {m.get('reference','')}]\n{d}"
        for d, m in zip(docs, metas)
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    sources = [{"title": m.get("title",""), "reference": m.get("reference",""),
                "url": m.get("source_url","")} for m in metas]
    return {"answer": answer, "sources": sources}



result = ask("Quelle est la définition du défaut selon Bâle ?")
print(result["answer"])
print("\nSources :")
for s in result["sources"]:
    print(f"  - {s['title'][:60]} ({s['reference']})")