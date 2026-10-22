# Step 1 : Charger les fichiers PDF et les convertir en pages
import os
from langchain_community.document_loaders import PyPDFLoader

base_dir = "data/processed"
all_pages = []

for folder in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(folder_path, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        all_pages.extend(pages)
        print(f"  {folder}/{filename} → {len(pages)} pages")

print(f"\nTotal : {len(all_pages)} pages chargées")
 # Step 2 : Découper les pages en chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,    # caractères (≈ 250-300 tokens)
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(all_pages)

print(f"Pages chargées  : {len(all_pages)}")
print(f"Chunks produits : {len(chunks)}")
print(f"\nExemple — chunk 0 :")
print(chunks[0].page_content[:300])
print(f"\nMétadonnées : {chunks[0].metadata}")


# Step 3 : Enrichir les chunks avec les métadonnées du corpus
import json

# Charger les métadonnées
with open("data/corpus_metadata.json", "r", encoding="utf-8") as f:
    corpus_metadata = json.load(f)

# Enrichir chaque chunk
not_found = set()

for chunk in chunks:
    source = chunk.metadata.get("source", "")
    # Normalise les backslashes Windows → forward slashes
    parts = source.replace("\\", "/").split("/")
    # source = "data/processed/bale_irb/fichier.pdf" → key = "bale_irb/fichier.pdf"
    try:
        idx = parts.index("processed")
        key = "/".join(parts[idx+1:idx+3])
    except ValueError:
        key = ""

    meta = corpus_metadata.get(key, {})
    if not meta:
        not_found.add(key)

    chunk.metadata.update({
        "document_id": meta.get("document_id", ""),
        "title":       meta.get("title", ""),
        "issuer":      meta.get("issuer", ""),
        "reference":   meta.get("reference", ""),
        "scope":       str(meta.get("scope", [])),   # ChromaDB n'accepte pas les listes
        "source_url":  meta.get("source_url", ""),
    })

print(f"Chunks enrichis : {len(chunks)}")
if not_found:
    print(f"Clés introuvables : {not_found}")
else:
    print("Toutes les clés matchées")

# Vérification sur le premier chunk
m = chunks[0].metadata
print(f"\nExemple :")
print(f"  document_id : {m['document_id']}")
print(f"  title       : {m['title'][:70]}")
print(f"  issuer      : {m['issuer']}")
print(f"  reference   : {m['reference']}")
# Step 4 : Indexer les chunks dans ChromaDB
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Client persistant — crée le dossier data/chroma_db
client = chromadb.PersistentClient(path="data/chroma_db")

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="regulatory_docs",
    embedding_function=embedding_fn
)

# Préparer les données
ids       = [f"chunk_{i}" for i in range(len(chunks))]
texts     = [c.page_content for c in chunks]
metadatas = [c.metadata for c in chunks]

# Nettoyer les None (ChromaDB n'accepte que str/int/float/bool)
for meta in metadatas:
    for k, v in meta.items():
        if v is None:
            meta[k] = ""

# Indexation par lots de 500
BATCH = 500
total = len(chunks)
print(f"Indexation de {total} chunks...")

for i in range(0, total, BATCH):
    collection.add(
        ids=ids[i:i+BATCH],
        documents=texts[i:i+BATCH],
        metadatas=metadatas[i:i+BATCH]
    )
    print(f"  {min(i+BATCH, total)}/{total}")

print(f"\nChromaDB : {collection.count()} chunks indexés")