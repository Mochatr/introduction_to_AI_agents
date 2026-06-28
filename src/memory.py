"""
Mémoire conversationnelle du graphe LangGraph.

LangGraph gère la persistance du state via un Checkpointer.
MemorySaver stocke chaque état en mémoire vive (idéal pour le dev/demo).
Chaque conversation est identifiée par un thread_id unique.
"""

from langgraph.checkpoint.memory import MemorySaver


def get_memory() -> MemorySaver:
    """Retourne un checkpointer en mémoire pour la persistance du state."""
    return MemorySaver()


def make_config(thread_id: str = "default") -> dict:
    """
    Génère la configuration LangGraph pour un thread de conversation.
    Le même thread_id permet à l'agent de se souvenir des échanges précédents.
    """
    return {"configurable": {"thread_id": thread_id}}
