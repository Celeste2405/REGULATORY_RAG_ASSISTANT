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