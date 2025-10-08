import os
import sys
import json
import chromadb
from langchain.schema import Document
# Add project root to sys.path for relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.text_embedder import TextEmbedder

class VectorDB:
    def __init__(self):
        self.data_path = os.path.join(project_root, "data", "processed", "university_docs.json")
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(project_root, "data" ,"chroma_db"))
        self.chroma_collection = None

    def load_chunk_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        return [Document(page_content=c["text"], metadata=c["metadata"]) for c in loaded_chunks]

    def create_collection(self, name="egyptian_public_universities"):
        text_embedder = TextEmbedder()
        embedding_function = text_embedder.embedding()
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )
        return self.chroma_collection

    def add_to_collection(self):
        if not self.chroma_collection:
            raise ValueError("No collection found")

        existing_ids = self.chroma_collection.get()["ids"]
        if not existing_ids:
            all_chunks = self.load_chunk_data()
            ids = [str(i) for i in range(len(all_chunks))]
            texts = [doc.page_content for doc in all_chunks]
            metadatas = [doc.metadata for doc in all_chunks]

            self.chroma_collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

        else:
            print("Collection already has data. Skipping add.")

    def search(self, query, k=5):
        if not self.chroma_collection:
            raise ValueError("No collection found")

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=k
        )
        return results
