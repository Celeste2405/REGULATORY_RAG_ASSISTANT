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