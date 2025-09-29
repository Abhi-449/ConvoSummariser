# rag_service.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def build_conversational_rag(vectorstore, llm):
    """
    Build a Conversational RAG chain with memory.
    """
    template = """
    You are a helpful assistant that summarizes and answers questions
    from documents and emails. Use both the retrieved context and the
    ongoing conversation to answer.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain
