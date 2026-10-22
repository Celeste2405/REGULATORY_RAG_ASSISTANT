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