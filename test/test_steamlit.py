import os
import pytest
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

@pytest.fixture(scope="module")
def collection():
    client = chromadb.PersistentClient(path="data/chroma_db")
    emb = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection("regulatory_docs", embedding_function=emb)

def test_collection_non_vide(collection):
    assert collection.count() > 10_000

def test_metadata_enrichie(collection):
    res = collection.query(query_texts=["défaut IRB"], n_results=5,
                           include=["metadatas"])
    for m in res["metadatas"][0]:
        assert m.get("document_id"), f"document_id absent : {m}"
        assert m.get("issuer"),      f"issuer absent : {m}"

def test_scope_stress_test(collection):
    res = collection.query(query_texts=["EBA stress test 2027"], n_results=5,
                           include=["metadatas"])
    scopes = [m.get("scope", "") for m in res["metadatas"][0]]
    assert any("STRESS_TEST" in s for s in scopes)

def test_scope_ifrs9(collection):
    res = collection.query(query_texts=["expected credit loss provisionnement"],
                           n_results=5, include=["metadatas"])
    scopes = [m.get("scope", "") for m in res["metadatas"][0]]
    assert any("IFRS9" in s for s in scopes)

def test_full_pipeline():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY absent")
    import sys; sys.path.insert(0, ".")
    from src.rag_chain import ask
    result = ask("Qu'est-ce que l'ICAAP ?", n=5)
    assert result["answer"]
    assert len(result["sources"]) > 0