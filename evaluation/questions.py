"""
20 questions d'evaluation du systeme Agentic RAG.
Domaine : Informatique / Intelligence Artificielle.

- 10 questions SIMPLES : faits directs, definitions, concepts de base.
- 10 questions COMPLEXES : analyse, comparaison, raisonnement multi-etapes.
"""

SIMPLE_QUESTIONS = [
    "What is a large language model (LLM)?",
    "What does RAG stand for and what is its purpose?",
    "What is prompt engineering?",
    "What is the transformer architecture?",
    "What is an embedding in the context of NLP?",
    "What is a vector database?",
    "What is the difference between fine-tuning and in-context learning?",
    "What is chain-of-thought prompting?",
    "What is the role of the retriever in a RAG system?",
    "What is an AI agent?",
]

COMPLEX_QUESTIONS = [
    "Compare the advantages and limitations of RAG systems versus fine-tuned models for domain-specific question answering.",
    "Explain how memory mechanisms in LLM-based agents differ from traditional database systems, and discuss their respective trade-offs.",
    "How does the self-attention mechanism in transformers enable long-range dependency modeling, and what are its computational limitations?",
    "Describe a multi-agent architecture for scientific research automation. What roles would each agent play and how would they coordinate?",
    "What strategies can be used to evaluate the quality of a RAG system, and what metrics are most relevant?",
    "How does chain-of-thought prompting improve reasoning in LLMs, and in what scenarios does it fail?",
    "Compare ReAct, Reflexion, and Tree of Thoughts as agent reasoning frameworks. When should each be preferred?",
    "What are the main challenges of deploying LLM agents in production environments, and how can they be mitigated?",
    "Explain how approximate nearest neighbor (ANN) search algorithms like HNSW or FAISS enable efficient semantic retrieval at scale.",
    "Design a complete Agentic RAG pipeline for a legal document analysis system. Justify your architectural choices.",
]

ALL_QUESTIONS = [
    {"id": i + 1, "type": "simple", "question": q}
    for i, q in enumerate(SIMPLE_QUESTIONS)
] + [
    {"id": i + 11, "type": "complex", "question": q}
    for i, q in enumerate(COMPLEX_QUESTIONS)
]
