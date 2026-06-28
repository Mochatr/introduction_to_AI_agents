"""
Étape 3 — Architecture du graphe LangGraph (Agentic RAG).

Flux :
  [START]
     |
  [retrieve]          : recherche sémantique dans ChromaDB
     |
  [grade]             : filtre les docs non pertinents
     |
  <decide_after_grade>  -- suffisant --> [generate] --> [END]
                        -- insuffisant --> [web_search] --> [generate] --> [END]
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.tools import retrieve_documents, grade_documents, web_search
from src.llm import get_llm
from src.memory import get_memory, make_config

load_dotenv()

# ── Prompt de génération ──────────────────────────────────────────────────────

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un assistant expert en informatique et intelligence artificielle. "
        "Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. "
        "Si le contexte est insuffisant, dis-le clairement. "
        "Réponds en français de manière structurée et précise.\n\n"
        "Contexte :\n{context}",
    ),
    ("human", "{question}"),
])

MAX_ITERATIONS = 3  # protection anti-boucle infinie

# ── Noeuds du graphe ──────────────────────────────────────────────────────────

def node_retrieve(state: AgentState) -> AgentState:
    """Noeud 1 : récupère les documents pertinents depuis ChromaDB."""
    question = state["question"]
    docs = retrieve_documents(question, k=5)
    return {
        **state,
        "retrieved_docs": [d.page_content for d in docs],
        "iterations": state.get("iterations", 0) + 1,
    }


def node_grade(state: AgentState) -> AgentState:
    """Noeud 2 : filtre les documents non pertinents via le LLM."""
    from langchain_core.documents import Document

    question = state["question"]
    raw_docs = [Document(page_content=c) for c in state["retrieved_docs"]]
    relevant = grade_documents(question, raw_docs)
    return {
        **state,
        "retrieved_docs": [d.page_content for d in relevant],
        "source": "vectorstore",  # sera ecrase si on passe par web_search
    }


def node_web_search(state: AgentState) -> AgentState:
    """Noeud 3 (fallback) : lance une recherche web si les docs locaux sont insuffisants."""
    question = state["question"]
    web_docs = web_search(question)
    extra = [d.page_content for d in web_docs]
    return {
        **state,
        "retrieved_docs": state["retrieved_docs"] + extra,
        "source": "web_search",
    }


def node_generate(state: AgentState) -> AgentState:
    """Noeud 4 : génère la réponse finale à partir des documents filtrés."""
    llm = get_llm()
    chain = _GENERATE_PROMPT | llm | StrOutputParser()

    context = "\n\n---\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else "Aucun document disponible."
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
    })

    messages = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=answer),
    ]

    return {
        **state,
        "answer": answer,
        "messages": messages,
    }


# ── Edges conditionnelles ─────────────────────────────────────────────────────

def decide_after_grade(state: AgentState) -> Literal["web_search", "generate"]:
    """
    Décision après le grading :
    - Si aucun doc pertinent ET iterations < MAX  → fallback web
    - Sinon → génération directe
    """
    no_docs = len(state["retrieved_docs"]) == 0
    too_many_iter = state.get("iterations", 0) >= MAX_ITERATIONS

    if no_docs and not too_many_iter:
        print("[GRAPH] Docs insuffisants -> web_search")
        return "web_search"

    print("[GRAPH] Docs suffisants -> generate")
    return "generate"


# ── Construction du graphe ────────────────────────────────────────────────────

def build_graph():
    """Construit et compile le graphe LangGraph avec mémoire."""
    memory = get_memory()

    builder = StateGraph(AgentState)

    # Ajout des noeuds
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("grade", node_grade)
    builder.add_node("web_search", node_web_search)
    builder.add_node("generate", node_generate)

    # Edges fixes
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)

    # Edge conditionnelle après le grading
    builder.add_conditional_edges(
        "grade",
        decide_after_grade,
        {"web_search": "web_search", "generate": "generate"},
    )

    return builder.compile(checkpointer=memory)


# ── Interface publique ────────────────────────────────────────────────────────

_graph = None

def get_graph():
    """Singleton : retourne le graphe compilé (construit une seule fois)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, thread_id: str = "default") -> str:
    """
    Point d'entrée principal : pose une question à l'agent.
    Le thread_id permet de maintenir le contexte conversationnel.
    """
    graph = get_graph()
    config = make_config(thread_id)

    initial_state: AgentState = {
        "messages": [],
        "question": question,
        "retrieved_docs": [],
        "answer": "",
        "source": "vectorstore",
        "iterations": 0,
    }

    result = graph.invoke(initial_state, config=config)
    return result["answer"], result.get("source", "vectorstore")


# ── Visualisation ASCII du graphe ─────────────────────────────────────────────

def visualize_graph():
    """Affiche la structure du graphe dans le terminal."""
    graph = get_graph()
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Visualisation indisponible : {e}")


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Visualisation du graphe ===")
    visualize_graph()

    print("\n=== Test question simple ===")
    answer = ask("What is an LLM agent?", thread_id="test-session")
    print(f"\nReponse:\n{answer}")

    print("\n=== Test question complexe ===")
    answer = ask(
        "Compare the memory mechanisms used in LLM agents with traditional database systems.",
        thread_id="test-session",
    )
    print(f"\nReponse:\n{answer}")
