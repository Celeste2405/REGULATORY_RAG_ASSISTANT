
import argparse
import random
from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chromadb"
COLLECTION = "regulatory_docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=5, help="nombre de chunks (défaut 5)")
    ap.add_argument("--doc", help="limiter à un document_id")
    args = ap.parse_args()

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    where = {"document_id": args.doc} if args.doc else None
    ids = col.get(where=where, include=[])["ids"]  # ids seuls : léger
    print(f"Total chunks dans ChromaDB : {col.count()}" + (f" ({len(ids)} pour {args.doc})" if args.doc else ""))
    if not ids:
        raise SystemExit("Aucun chunk.")

    res = col.get(ids=random.sample(ids, min(args.n, len(ids))), include=["documents", "metadatas"])
    print("=" * 60)
    for i, (doc, m) in enumerate(zip(res["documents"], res["metadatas"]), 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Document  : {m.get('document_id')}")
        print(f"Titre     : {m.get('title')}")
        print(f"Reference : {m.get('reference') or '-'}")
        print(f"Regulator : {m.get('regulator')}")
        print(f"Date      : {m.get('date') or '-'}")
        print(f"Article   : {m.get('article') or '-'}")
        print(f"Page      : {m.get('page')}")
        print(f"Longueur  : {len(doc)} car.")
        print(f"Texte     : {doc[:300]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()
