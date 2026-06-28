"""Définition du state partagé dans le graphe LangGraph."""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State central du graphe Agentic RAG."""

    # Historique des messages (accumulé via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # Question posée par l'utilisateur
    question: str

    # Documents récupérés par le retriever
    retrieved_docs: list[str]

    # Réponse finale générée
    answer: str

    # Source utilisée : "vectorstore" ou "web_search"
    source: str

    # Nombre d'itérations (protection anti-boucle infinie)
    iterations: int
