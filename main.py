from ingestion_service import IngestionService
from vector_store import VectorStore
from llm_service import load_llm
from rag_service import build_conversational_rag

if __name__ == "__main__":
    # Step 1: scan + ingest
    ingestor = IngestionService("data")
    files = ingestor.scan_folder()
    print(f"[INFO] Found {len(files)} files: {files}")

    chunks = ingestor.ingest(files)
    print(f"[INFO] Ingested {len(chunks)} chunks")

    # Step 2: build FAISS index
    store = VectorStore()                  # ✅ no chunks here
    store.build_index(chunks)              # ✅ chunks go here

    # Step 3: load LLM
    llm = load_llm("google/flan-t5-base")

    # Step 4: build conversational RAG
    qa_chain = build_conversational_rag(store.vectorstore, llm)

    # Step 5: conversation loop
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.run({"question": query})
        print("\nBot:", response)
