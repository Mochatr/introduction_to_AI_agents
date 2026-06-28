"""
Étape 2 — Outils de l'agent.

Trois outils sont exposés au graphe LangGraph :
  1. retrieve_documents  : recherche sémantique dans ChromaDB
  2. grade_documents     : filtre les chunks non pertinents via le LLM
  3. web_search          : fallback si le vectorstore est insuffisant
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

from src.llm import get_llm
from src.ingestion import load_vectorstore

# ── 1. Retriever ──────────────────────────────────────────────────────────────

def retrieve_documents(question: str, k: int = 5) -> List[Document]:
    """
    Recherche les k documents les plus similaires à la question
    dans la base vectorielle ChromaDB.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    print(f"[TOOL] retrieve_documents -> {len(docs)} docs recuperes.")
    return docs


# ── 2. Grader de pertinence ───────────────────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un évaluateur de pertinence documentaire. "
        "Réponds uniquement par 'yes' si le document est pertinent "
        "pour répondre à la question, ou 'no' sinon. "
        "Pas d'explication.",
    ),
    (
        "human",
        "Question : {question}\n\nDocument :\n{document}",
    ),
])


def grade_documents(question: str, docs: List[Document]) -> List[Document]:
    """
    Filtre les documents récupérés : ne garde que ceux jugés pertinents.
    Evalue les 3 premiers docs max pour limiter la consommation de tokens.
    """
    llm = get_llm()
    chain = _GRADE_PROMPT | llm | StrOutputParser()

    relevant = []
    for doc in docs[:3]:  # max 3 docs evalues pour economiser les tokens
        score = chain.invoke({
            "question": question,
            "document": doc.page_content[:600],  # extrait reduit
        }).strip().lower()
        if score == "yes":
            relevant.append(doc)

    # Ajouter les docs restants sans grading si on a deja des resultats
    if relevant and len(docs) > 3:
        relevant.extend(docs[3:])

    print(f"[TOOL] grade_documents -> {len(relevant)}/{len(docs)} docs pertinents.")
    return relevant


# ── 3. Web search (fallback) ──────────────────────────────────────────────────

_search_engine = DuckDuckGoSearchRun()


def web_search(question: str) -> List[Document]:
    """
    Lance une recherche web via DuckDuckGo et retourne les résultats
    sous forme de Documents LangChain (fallback quand le vectorstore
    ne contient pas de réponse pertinente).
    """
    print(f"[TOOL] web_search -> requete : '{question}'")
    try:
        raw = _search_engine.invoke(question)
        doc = Document(
            page_content=raw,
            metadata={"source": "web_search", "query": question},
        )
        return [doc]
    except Exception as e:
        print(f"[WARN] web_search échec : {e}")
        return []


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = "What is Retrieval-Augmented Generation?"

    print("\n=== Test retrieve_documents ===")
    docs = retrieve_documents(question, k=4)

    print("\n=== Test grade_documents ===")
    relevant = grade_documents(question, docs)
    for i, d in enumerate(relevant, 1):
        print(f"\n[Doc {i}] {d.metadata.get('source', 'N/A')}")
        print(d.page_content[:200])

    print("\n=== Test web_search ===")
    web_docs = web_search(question)
    for d in web_docs:
        print(d.page_content[:300])
