# vector_store.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingestion_service import IngestionService  # import our ingestion module


class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def build_index(self, docs, persist_path="faiss_index"):
        self.vectorstore = FAISS.from_documents(docs, self.embedder)
        self.vectorstore.save_local(persist_path)
        print(f"✅ FAISS index built & saved at {persist_path}")

    def load_index(self, persist_path="faiss_index"):
        self.vectorstore = FAISS.load_local(persist_path, self.embedder)
        print(f"✅ FAISS index loaded from {persist_path}")

    def search(self, query, k=3):
        if not self.vectorstore:
            raise ValueError("Index not loaded. Build or load index first.")
        return self.vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    # 🔹 Example usage
    files = ["sample.pdf", "notes.txt", "email.eml"]  # make sure these exist
    ingestor = IngestionService()
    chunks = ingestor.ingest(files)   # <-- chunks defined HERE

    store = VectorStore()
    store.build_index(chunks)

    # 🔎 test query
    results = store.search("What is the attendace percentage of abhishek?", k=2)
    for r in results:
        print("\n--- Result ---")
        print("Content:", r.page_content[:200])
        print("Metadata:", r.metadata)
