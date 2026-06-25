"""
Vector memory for the weather agent.

This is the same idea as Milestone 3 of the notebook (ChromaDB +
VectorStoreRetrieverMemory): store each conversation turn as an embedding,
then retrieve only the turns whose *meaning* is similar to the new question.

We embed with the local Ollama model `nomic-embed-text` and keep the vectors
in a small JSON file, so there is no chromadb dependency. On Python 3.14 pip
backtracks to older chromadb releases that pull in pandas, and that pandas has
no 3.14 wheel so it tries to build from source and fails (setuptools 81+ removed
`pkg_resources`). Rather than fight that, we reimplement the slice we use. The
behaviour is identical:

    save_context(...)         → embed the turn, persist it
    recall("where do I live") → cosine-similarity search, return the closest turns

It survives restarts because everything is written to `store_path`.
"""

import json
import math
import os

import ollama

EMBED_MODEL = "nomic-embed-text"


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorMemory:
    """A tiny persistent vector store — what ChromaDB gives you, in ~40 lines."""

    def __init__(self, store_path="./weather_memory.json", k=3, min_score=0.55):
        self.store_path = store_path
        self.k = k                 # how many past turns to retrieve
        self.min_score = min_score # ignore weakly-related memories
        self.items = []            # list of {"text": str, "embedding": [float]}
        if os.path.exists(store_path):
            with open(store_path) as f:
                self.items = json.load(f)

    def _embed(self, text):
        return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

    def _persist(self):
        with open(self.store_path, "w") as f:
            json.dump(self.items, f)

    def save_context(self, user_text, agent_text=""):
        """Store one conversation turn (the user line is what we search on)."""
        text = f"User said: {user_text}"
        if agent_text:
            text += f" | Assistant replied: {agent_text}"
        self.items.append({"text": text, "embedding": self._embed(text)})
        self._persist()

    def recall(self, query):
        """Return up to k past turns most relevant to `query`, best first."""
        if not self.items:
            return []
        q = self._embed(query)
        scored = [(_cosine(q, it["embedding"]), it["text"]) for it in self.items]
        scored.sort(reverse=True)
        return [text for score, text in scored[: self.k] if score >= self.min_score]

    def count(self):
        return len(self.items)

    def clear(self):
        self.items = []
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
