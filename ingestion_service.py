import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredEmailLoader

class IngestionService:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    def scan_folder(self):
        """Return all supported files in the folder"""
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith((".pdf", ".txt", ".eml"))
        ]

    def load_file(self, path):
        """Load a single file and return a list of LangChain Document objects"""
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".eml"):
            loader = UnstructuredEmailLoader(path)
        else:
            print(f"[WARN] Unsupported file type: {path}")
            return []

        return loader.load()  # returns a list of Document objects

    def ingest(self, files):
        all_chunks = []
        for path in files:
            docs = self.load_file(path)
            if docs:
                chunks = self.splitter.split_documents(docs)
                all_chunks.extend(chunks)
        return all_chunks
