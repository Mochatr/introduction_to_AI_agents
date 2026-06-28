# Agentic RAG — Système RAG Agentique avec LangGraph

Projet de fin de module — Master IIBDCC (SMA et IAD)  
Auteur : Mohamed CHATR

---

## Présentation

Ce projet implémente un **système RAG Agentique complet** (Retrieval-Augmented Generation) en utilisant **LangGraph** pour construire manuellement le graphe de raisonnement de l'agent. Contrairement à `create_agent` de LangChain qui fournit un agent pré-construit, ce projet définit explicitement chaque nœud, chaque transition et la logique de décision de l'agent.

Le système est capable de répondre à des questions complexes dans le domaine de l'**Informatique et de l'Intelligence Artificielle**, en s'appuyant sur une base documentaire vectorisée et un mécanisme de fallback vers la recherche web.

---

## Architecture du graphe

```
[START]
   │
[retrieve]        ← Recherche sémantique dans ChromaDB
   │
[grade]           ← Filtrage des documents non pertinents (LLM)
   │
   ├── docs pertinents ──→ [generate] ──→ [END]
   │
   └── aucun doc ────────→ [web_search] ──→ [generate] ──→ [END]
```

### Nœuds du graphe

| Nœud | Rôle |
|---|---|
| `retrieve` | Interroge ChromaDB via similarité sémantique (top-k=5) |
| `grade` | Demande au LLM d'évaluer la pertinence de chaque document |
| `web_search` | Fallback DuckDuckGo si aucun document local n'est pertinent |
| `generate` | Génère la réponse finale en français à partir du contexte filtré |

---

## Stack technique

| Composant | Technologie |
|---|---|
| Orchestration agentique | LangGraph 1.2.6 |
| LLM | Groq API — Llama 3.3 70b Versatile |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Base vectorielle | ChromaDB (persistance locale) |
| Chargement documents | LangChain PyPDFLoader + WebBaseLoader |
| Recherche web (fallback) | DuckDuckGo Search |
| Mémoire conversationnelle | LangGraph MemorySaver (thread_id) |
| Langage | Python 3.12 |

---

## Structure du projet

```
agentic_AI/
│
├── .env                        
├── requirements.txt            # Dépendances Python
├── main.py                     # Point d'entrée CLI
│
├── data/                       # Documents PDF sources
│
├── vectorstore/                # Base ChromaDB persistée
│
├── src/
│   ├── llm.py                  # Initialisation LLM Groq + embeddings
│   ├── state.py                # AgentState (TypedDict LangGraph)
│   ├── ingestion.py            # Chargement, chunking, vectorisation
│   ├── tools.py                # Outils : retrieve, grade, web_search
│   ├── graph.py                # Construction et compilation du graphe
│   └── memory.py               # MemorySaver + configuration thread_id
│
└── evaluation/
    ├── questions.py            # 10 questions simples + 10 complexes
    ├── evaluate.py             # Script d'évaluation automatique
    └── results.csv             # Résultats générés
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/agentic_AI.git
cd agentic_AI
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine :

```env
GROQ_API_KEY=your_groq_api_key_here
USER_AGENT=AgenticRAG/1.0
```

> Clé Groq gratuite disponible sur [console.groq.com](https://console.groq.com)

---

## Utilisation

### Construire le vectorstore (première fois)

```bash
python main.py --build
```

### Mode interactif (conversation)

```bash
python main.py
```

```
============================================================
  Agentic RAG — Domaine : Informatique / IA
  Tape 'quit' ou 'exit' pour terminer.
============================================================

Question : What is a RAG system?

Recherche en cours...
------------------------------------------------------------
Un système RAG (Retrieval-Augmented Generation) est...
------------------------------------------------------------
```

### Visualiser le graphe LangGraph

```bash
python main.py --visualize
```

### Lancer l'évaluation complète (20 questions)

```bash
python main.py --evaluate
```

---

## Base documentaire

Les sources suivantes sont chargées automatiquement depuis le blog [Lilian Weng](https://lilianweng.github.io) :

| Article | Thème |
|---|---|
| LLM Powered Autonomous Agents | Agents IA, mémoire, planification, outils |
| Prompt Engineering | Techniques de prompting, CoT, ReAct |
| Deep Learning Recommendation Model | Apprentissage profond |
| The Transformer Family v2 | Architecture Transformer, attention |
| How to Train Large Neural Networks | Entraînement à grande échelle |

Des PDFs supplémentaires peuvent être placés dans le dossier `data/` , ils seront automatiquement ingérés au prochain `--build`.

---

## Évaluation du système

Le système a été évalué sur **20 questions** (10 simples + 10 complexes) :

| Catégorie | Temps moyen | Score moyen (/5) |
|---|---|---|
| Questions simples | 14.5s | 5.0 |
| Questions complexes | 18.6s | 4.9 |
| **Global** | **16.5s** | **4.95** |

**Répartition des sources :**
- Vectorstore local : 13/20 questions (65%)
- Fallback web search : 7/20 questions (35%)

Les résultats détaillés sont disponibles dans [`evaluation/results.csv`](evaluation/results.csv).

---

## Fonctionnement de la mémoire

L'agent maintient un historique conversationnel via le système de **checkpointing LangGraph**. Chaque session est identifiée par un `thread_id`, le même identifiant permet à l'agent de se souvenir des échanges précédents au sein d'une même conversation.

```python
from src.graph import ask

# Même thread_id = l'agent se souvient du contexte
answer1 = ask("What is an LLM agent?", thread_id="session-1")
answer2 = ask("Can you elaborate on its memory component?", thread_id="session-1")
```

---

## Limites et pistes d'amélioration

- **Base documentaire limitée** : 5 articles web seulement. enrichissez avec des PDFs ArXiv.
- **Embeddings mono-langue** : `all-MiniLM-L6-v2` est optimisé pour l'anglais. Utilisez un modèle multilingue pour des questions en français
- **Pas de reranking** : ajoutez un cross-encoder pour affiner le classement des documents récupérés
- **Évaluation manuelle** : le score LLM auto-évalué peut être biaisé. intégrez RAGAS pour une évaluation plus rigoureuse
- **Scalabilité** : ChromaDB local peut être remplacé par Pinecone ou Qdrant pour un déploiement en production

---