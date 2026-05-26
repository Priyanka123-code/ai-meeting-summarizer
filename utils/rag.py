import json
import os

import streamlit as st


# ====================== RAG VECTOR STORE ======================
@st.cache_resource
def get_vectorstore():
    """Returns the Chroma vector store when optional RAG packages are installed."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory="data/vector_db",
        embedding_function=embeddings,
        collection_name="meeting_transcripts",
    )


def add_to_rag(transcript_data: dict, analysis):
    """
    Add a processed meeting to the optional vector database.
    Lightweight deployments skip vector indexing and keep the JSON library.
    """
    try:
        from langchain_core.documents import Document

        vectorstore = get_vectorstore()

        doc_content = f"""Meeting Title: {transcript_data['filename']}
Date: {transcript_data['date']}
Summary: {analysis.summary}
Transcript: {transcript_data['transcript']}"""

        doc = Document(
            page_content=doc_content,
            metadata={
                "meeting_id": transcript_data["meeting_id"],
                "filename": transcript_data["filename"],
                "date": transcript_data["date"],
            },
        )

        vectorstore.add_documents([doc])
        st.success("Meeting added to RAG Knowledge Base!")
    except ImportError:
        st.info("Lightweight deployment: meeting saved to the library without vector indexing.")


def _simple_library_search(query: str, k: int = 3):
    transcript_dir = "data/transcripts"
    terms = [term.lower() for term in query.split() if len(term) > 2]
    matches = []

    if not os.path.exists(transcript_dir):
        return [], ""

    for filename in os.listdir(transcript_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(transcript_dir, filename)
        try:
            with open(path, "r") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        text = " ".join(
            [
                data.get("filename", ""),
                data.get("date", ""),
                data.get("transcript", ""),
                data.get("analysis", {}).get("summary", ""),
            ]
        )
        score = sum(text.lower().count(term) for term in terms)
        if score:
            matches.append((score, data))

    matches.sort(key=lambda item: item[0], reverse=True)
    selected = [data for _, data in matches[:k]]
    context = "\n\n".join(
        f"Meeting Title: {data.get('filename', 'Unknown')}\n"
        f"Date: {data.get('date', 'Unknown')}\n"
        f"Summary: {data.get('analysis', {}).get('summary', '')}\n"
        f"Transcript: {data.get('transcript', '')}"
        for data in selected
    )

    return selected, context


def rag_search(query: str, k: int = 3):
    """
    Search across all past meetings.
    Uses vector search when installed, otherwise falls back to simple library search.
    """
    if not query:
        return [], ""

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return docs, context
    except ImportError:
        return _simple_library_search(query, k)
