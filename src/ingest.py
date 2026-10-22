"""Ingestion : PDF -> chunks (512/64) -> embeddings all-MiniLM-L6-v2 -> ChromaDB persistante.

Usage :
    python src/ingest.py            # indexe data/processed/**/*.pdf
    python src/ingest.py --reset    # supprime la collection avant de réindexer
"""
import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

import chromadb
import pymupdf as fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paramètres du projet : chemins, collection ChromaDB, modèle d'embeddings, taille des chunks, batch d'indexation
ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chromadb"
COLLECTION = "regulatory_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 512, 64
BATCH = 64

# "Article 178", "Art. 12a", "Article premier" en début de ligne = titre d'article
ARTICLE_HEADING = re.compile(r"^\s*(?:Article|Art\.)\s+(\d+[a-z]?)\b", re.M | re.I)
ARTICLE_ANY = re.compile(r"\b(?:Article|Art\.)\s+(\d+[a-z]?)\b", re.I)
FILENAME_YEAR = re.compile(r"(?:^|_)((?:19|20)\d{2})(?:_|\.|$)")


def doc_date(meta: dict, path: Path) -> str:
    """Date du document : métadonnée PDF (creationDate 'D:YYYYMMDD...'), sinon année du nom de fichier."""
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", (meta or {}).get("creationDate") or "")
    if m:
        return "-".join(m.groups())
    m = FILENAME_YEAR.search(path.name)
    return m.group(1) if m else ""


def load_pdf(path: Path):
    """Retourne (pages[(num, texte)], date). PyMuPDF : ~100x plus rapide que PyPDF2."""
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((i, text))
        date = doc_date(pdf.metadata, path)
    return pages, date


def chunk_document(path: Path, splitter: RecursiveCharacterTextSplitter):
    pages, date = load_pdf(path)
    domain = path.parent.name
    source = path.name.removesuffix(".pdf").removesuffix(".pdf")  # fichiers en .pdf.pdf
    current_article = ""
    ids, docs, metas = [], [], []
    for page_num, text in pages:
        for j, chunk in enumerate(splitter.split_text(text)):
            heading = ARTICLE_HEADING.search(chunk) or ARTICLE_ANY.search(chunk)
            if heading:
                current_article = f"Article {heading.group(1)}"
            ids.append(hashlib.md5(f"{path.name}|{page_num}|{j}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({
                "source": source,
                "article": current_article,
                "date": date,
                "domain": domain,
                "page": page_num,
            })
    return ids, docs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="vide la collection avant indexation")
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Aucun PDF trouvé dans {PDF_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    model = SentenceTransformer(args.model)
    model.max_seq_length = 256  # 512 car. ~ 130-170 tokens ; évite le padding inutile
    prefix = "passage: " if "e5" in args.model else ""  # convention E5

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    col = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

    total = 0
    for pdf in tqdm(pdfs, desc="PDF"):
        ids, docs, metas = chunk_document(pdf, splitter)
        for k in range(0, len(docs), BATCH):
            b = slice(k, k + BATCH)
            # reprise : on saute les chunks déjà indexés (ids déterministes)
            done = set(col.get(ids=ids[b], include=[])["ids"])
            if len(done) == len(ids[b]):
                continue
            todo = [n for n, i in enumerate(ids[b]) if i not in done]
            ids_b = [ids[b][n] for n in todo]
            docs_b = [docs[b][n] for n in todo]
            metas_b = [metas[b][n] for n in todo]
            emb = model.encode([prefix + d for d in docs_b], normalize_embeddings=True).tolist()
            col.upsert(ids=ids_b, documents=docs_b, embeddings=emb, metadatas=metas_b)
        total += len(docs)
    print(f"{total} chunks indexés ({col.count()} au total) dans {CHROMA_DIR} "
          f"[{datetime.now():%H:%M:%S}]")


if __name__ == "__main__":
    main()
