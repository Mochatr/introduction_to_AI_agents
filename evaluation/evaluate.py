"""
Etape 4 — Evaluation du systeme Agentic RAG.

Pour chaque question (10 simples + 10 complexes), on mesure :
  - Temps de reponse (secondes)
  - Longueur de la reponse (nombre de mots)
  - Score de pertinence auto-evalue par le LLM (1-5)
  - Source utilisee (vectorstore ou web_search)

Resultats exportes dans evaluation/results.csv et affiches en console.
"""

import time
import csv
import os
from pathlib import Path
import random

from src.graph import ask
from src.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from evaluation.questions import ALL_QUESTIONS

RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"

# Prompt pour auto-evaluer la qualite d'une reponse
_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un evaluateur expert. Note la reponse suivante de 1 a 5 "
        "selon sa pertinence, completude et exactitude par rapport a la question. "
        "Reponds UNIQUEMENT avec un chiffre entre 1 et 5.",
    ),
    ("human", "Question : {question}\n\nReponse : {answer}"),
])


def score_answer(question: str, answer: str) -> int:
    """Demande au LLM de noter la reponse de 1 a 5. Retry si rate limit."""
    llm = get_llm()
    chain = _EVAL_PROMPT | llm | StrOutputParser()
    for attempt in range(3):
        try:
            raw = chain.invoke({"question": question, "answer": answer}).strip()
            return int(raw[0])
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 60 * (attempt + 1)
                safe_print(f"     [Rate limit] Pause {wait}s avant retry...")
                time.sleep(wait)
            else:
                return 0
    return 0


def safe_print(text: str):
    """Print qui ne plante pas sur les terminaux Windows cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def run_evaluation():
    """Lance l'evaluation complete sur les 20 questions et exporte les resultats."""
    safe_print("\n" + "=" * 65)
    safe_print("  EVALUATION DU SYSTEME AGENTIC RAG")
    safe_print("  20 questions (10 simples + 10 complexes)")
    safe_print("=" * 65)

    rows = []

    for entry in ALL_QUESTIONS:
        qid = entry["id"]
        qtype = entry["type"]
        question = entry["question"]

        safe_print(f"\n[Q{qid:02d}] ({qtype.upper()}) {question[:70]}...")

        # Mesure du temps
        start = time.time()
        answer, source = ask(question, thread_id=f"eval-{qid}")
        elapsed = round(time.time() - start, 2)

        # Metriques
        word_count = len(answer.split())
        quality_score = score_answer(question, answer)

        safe_print(f"     Temps : {elapsed}s | Mots : {word_count} | Score : {quality_score}/5 | Source : {source}")

        # Pause entre questions pour eviter le rate limit Groq (100k TPD)
        time.sleep(5)

        rows.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "answer": answer,
            "response_time_s": elapsed,
            "word_count": word_count,
            "quality_score": quality_score,
            "source": source,
        })

    # Export CSV
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Resume final
    safe_print("\n" + "=" * 65)
    safe_print("  RESUME")
    safe_print("=" * 65)

    simple = [r for r in rows if r["type"] == "simple"]
    complex_ = [r for r in rows if r["type"] == "complex"]

    def avg(lst, key):
        return round(sum(r[key] for r in lst) / len(lst), 2) if lst else 0

    safe_print(f"  Questions simples  | Temps moy : {avg(simple, 'response_time_s')}s "
               f"| Score moy : {avg(simple, 'quality_score')}/5")
    safe_print(f"  Questions complexes| Temps moy : {avg(complex_, 'response_time_s')}s "
               f"| Score moy : {avg(complex_, 'quality_score')}/5")
    safe_print(f"  Score global       : {avg(rows, 'quality_score')}/5")
    safe_print(f"\n  Resultats exportes : {RESULTS_PATH}")
    safe_print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
