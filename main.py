"""
Point d'entrée principal — interface CLI du système Agentic RAG.

Usage :
  python main.py                  # mode interactif (conversation)
  python main.py --build          # (re)construit le vectorstore
  python main.py --visualize      # affiche le graphe LangGraph
  python main.py --evaluate       # lance l'evaluation sur les 20 questions
"""

import argparse
import os
import sys
import io

# Force UTF-8 sur Windows (évite les UnicodeEncodeError en console cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "AgenticRAG/1.0")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def cmd_build():
    from src.ingestion import get_or_build_vectorstore
    get_or_build_vectorstore()
    print("Vectorstore pret.")


def cmd_visualize():
    from src.graph import visualize_graph
    visualize_graph()


def cmd_evaluate():
    from evaluation.evaluate import run_evaluation
    run_evaluation()


def cmd_interactive():
    from src.graph import ask

    print("=" * 60)
    print("  Agentic RAG — Domaine : Informatique / IA")
    print("  Tape 'quit' ou 'exit' pour terminer.")
    print("=" * 60)

    thread_id = "interactive-session"
    while True:
        try:
            question = input("\nQuestion : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Au revoir.")
            break

        print("\nRecherche en cours...\n")
        answer, source = ask(question, thread_id=thread_id)
        print("-" * 60)
        print(answer)
        print(f"\n[Source : {source}]")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG — LangGraph")
    parser.add_argument("--build", action="store_true", help="Construit le vectorstore")
    parser.add_argument("--visualize", action="store_true", help="Affiche le graphe")
    parser.add_argument("--evaluate", action="store_true", help="Lance l'evaluation")
    args = parser.parse_args()

    if args.build:
        cmd_build()
    elif args.visualize:
        cmd_visualize()
    elif args.evaluate:
        cmd_evaluate()
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
