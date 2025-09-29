import streamlit as st
from tempfile import NamedTemporaryFile
import os
from ingestion_service import IngestionService
from vector_store import VectorStore
from llm_service import load_llm
from rag_service import build_conversational_rag


st.set_page_config(page_title="ConvoSummarizer", layout="wide")
st.title("📝 ConvoSummarizer - RAG Chatbot")

# --- Initialize LLM ---
@st.cache_resource
def init_llm():
    return load_llm("google/flan-t5-base")

llm = init_llm()
qa_chain = None
vectorstore = VectorStore()

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload documents (PDF, TXT, EML)", accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing uploaded files...")
    temp_docs = []

    for uploaded_file in uploaded_files:
        # Save to temp file for ingestion
        with NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        ingestor = IngestionService(os.path.dirname(tmp_path))
        docs = ingestor.ingest([tmp_path])
        temp_docs.extend(docs)


    st.success(f"Ingested {len(temp_docs)} chunks from uploaded files.")

    # Build FAISS index
    vectorstore.build_index(temp_docs)
    qa_chain = build_conversational_rag(vectorstore.vectorstore, llm)
    st.success("Vectorstore and RAG chain ready. You can now ask questions!")

# --- Chat input ---
if qa_chain:
    user_input = st.text_input("Ask something about your uploaded documents:")
    if user_input:
        response = qa_chain.run({"question": user_input})
        st.markdown(f"**🤖 Bot:** {response}")
