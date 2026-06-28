"""
Étape 1 — Ingestion documentaire.
Charge les documents (PDF locaux ou pages web), les découpe en chunks
et les vectorise dans une base ChromaDB persistante.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.llm import get_embeddings

# Répertoires
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"
COLLECTION_NAME = "agentic_rag_informatique"

# Sources web (articles IA/Informatique libres de droit)
WEB_SOURCES = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2020-10-29-dlrm/",
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
    "https://lilianweng.github.io/posts/2021-09-25-train-large/",
]


def load_pdfs() -> List[Document]:
    """Charge tous les PDFs présents dans data/."""
    docs = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("[INFO] Aucun PDF trouvé dans data/ — seules les sources web seront utilisées.")
        return docs
    for pdf_path in pdf_files:
        print(f"[INFO] Chargement PDF : {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"[INFO] {len(docs)} pages PDF chargées.")
    return docs


def load_web_sources() -> List[Document]:
    """Charge les articles depuis les URLs définies."""
    docs = []
    for url in WEB_SOURCES:
        try:
            print(f"[INFO] Chargement web : {url}")
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Échec chargement {url} : {e}")
    print(f"[INFO] {len(docs)} documents web chargés.")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Découpe les documents en chunks chevauchants."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] {len(chunks)} chunks créés.")
    return chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Crée et persiste la base vectorielle ChromaDB."""
    print("[INFO] Vectorisation et indexation en cours…")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"[INFO] Vectorstore créé : {len(chunks)} chunks indexés dans '{VECTORSTORE_DIR}'.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Charge une base ChromaDB existante."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    return vectorstore


def get_or_build_vectorstore() -> Chroma:
    """
    Retourne le vectorstore existant s'il est déjà construit,
    sinon le crée à partir des sources disponibles.
    """
    chroma_data = VECTORSTORE_DIR / "chroma.sqlite3"
    if chroma_data.exists():
        print("[INFO] Vectorstore existant détecté — chargement direct.")
        return load_vectorstore()

    print("[INFO] Construction du vectorstore depuis zéro…")
    docs = load_pdfs() + load_web_sources()
    if not docs:
        raise ValueError("Aucun document disponible. Ajoutez des PDFs dans data/ ou vérifiez les URLs.")
    chunks = split_documents(docs)
    return build_vectorstore(chunks)


if __name__ == "__main__":
    vs = get_or_build_vectorstore()
    # Test rapide : recherche sémantique
    results = vs.similarity_search("What is a RAG system?", k=3)
    print("\n--- Test de récupération ---")
    for i, doc in enumerate(results, 1):
        print(f"\n[Doc {i}] Source: {doc.metadata.get('source', 'N/A')}")
        print(doc.page_content[:300])
