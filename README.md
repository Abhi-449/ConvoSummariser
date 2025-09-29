# ConvoSummariser
ConvoSummarizer is a Retrieval-Augmented Generation (RAG) chatbot that can ingest documents (PDF, TXT, EML), build embeddings, and answer user questions about them. Built with LangChain, HuggingFace Transformers, FAISS, and Streamlit, it’s designed as a modular prototype that runs on a local machine (even on modest GPUs like RTX 3050).

🚀 Features

📂 Upload documents directly (PDF, TXT, EML).

🔍 Automatic ingestion + text chunking.

🧠 Embeddings with FAISS vector search.

🤖 Question answering using HuggingFace LLMs (Flan-T5).

🎛️ Streamlit interface with file upload + chatbot.

⚡ Runs locally with no paid API keys required.

🛠️ Tech Stack

LangChain → document loaders, RAG pipeline.

HuggingFace Transformers → Flan-T5 model for text generation.

FAISS → vectorstore for retrieval.

Streamlit → frontend for chatbot + file upload.

Python → core backend services.

📂 Project Structure
Convosummarizer/
│── app.py                  # Streamlit UI
│── ingestion_service.py    # File ingestion + chunking
│── embedding_service.py    # Embedding + FAISS index
│── llm_service.py          # HuggingFace LLM loader
│── rag_service.py          # RAG chain builder
│── data/                   # Sample documents
│── requirements.txt        # Dependencies

⚡ Quickstart

Clone the repo

git clone https://github.com/<your-username>/Convosummarizer.git
cd Convosummarizer


Create virtual environment

conda create -n convosummarizer python=3.10 -y
conda activate convosummarizer


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


Upload files & chat

Drop in your .pdf, .txt, or .eml files.

Ask questions → get answers from your documents.

📌 Roadmap

 Add conversation memory (multi-turn chats).

 Chat-style UI with st.chat_message.

 Persistent vectorstore (save embeddings across sessions).

 Deploy frontend in React for multi-tech stack showcase.
