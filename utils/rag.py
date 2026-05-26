import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# ====================== RAG VECTOR STORE ======================
@st.cache_resource
def get_vectorstore():
    """Returns the Chroma vector store (cached for performance)"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(
        persist_directory="data/vector_db",
        embedding_function=embeddings,
        collection_name="meeting_transcripts"
    )

def add_to_rag(transcript_data: dict, analysis):
    """
    Add a processed meeting to the RAG vector database
    """
    vectorstore = get_vectorstore()

    doc_content = f"""Meeting Title: {transcript_data['filename']}
Date: {transcript_data['date']}
Summary: {analysis.summary}
Transcript: {transcript_data['transcript']}"""

    doc = Document(
        page_content=doc_content,
        metadata={
            "meeting_id": transcript_data['meeting_id'],
            "filename": transcript_data['filename'],
            "date": transcript_data['date']
        }
    )

    vectorstore.add_documents([doc])
    st.success("✅ Meeting added to RAG Knowledge Base!")

def rag_search(query: str, k: int = 3):
    """
    Search across all past meetings using RAG
    """
    if not query:
        return [], ""

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return docs, context