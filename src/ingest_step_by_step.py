import json

with open("data/corpus_metadata.json") as f:
    corpus_meta = json.load(f)

# Afficher les clés (noms de fichiers)
for filename, meta in corpus_meta.items():
    print(filename, "→", meta["reference"])


