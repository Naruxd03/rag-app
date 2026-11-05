"""Minimal RAG app: load -> chunk -> embed -> store -> retrieve -> generate."""
import json
import pathlib
import re
import sys

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# ---------- Settings (report these in your results note) ----------
DOCS_DIR = "docs"
INDEX_FILE = "index.npy"      # saved embeddings
CHUNKS_FILE = "chunks.json"   # saved chunk text + source info
CHUNK_WORDS = 150             # words per chunk
OVERLAP = 30                  # words shared between neighbouring chunks
TOP_K = 4
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
# -------------------------------------------------------------------

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# Step 2: load and clean
def load_documents():
    docs = []
    for path in sorted(pathlib.Path(DOCS_DIR).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank lines
        text = re.sub(r"[ \t]+", " ", text).strip()  # collapse spaces
        docs.append((path.stem, text))               # keep the source name
    return docs


# Step 3: chunk with overlap
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# Step 4: embed and store
def build_index():
    chunks = []
    for name, text in load_documents():
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({"source": name, "chunk_id": i, "text": piece})
    print(f"Created {len(chunks)} chunks. Embedding...")
    vectors = get_embedder().encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,  # so dot product = cosine similarity
        show_progress_bar=True,
    )
    np.save(INDEX_FILE, vectors)
    pathlib.Path(CHUNKS_FILE).write_text(json.dumps(chunks), encoding="utf-8")
    return vectors, chunks


def load_index():
    if not (pathlib.Path(INDEX_FILE).exists() and pathlib.Path(CHUNKS_FILE).exists()):
        return build_index()
    vectors = np.load(INDEX_FILE)
    chunks = json.loads(pathlib.Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    return vectors, chunks


# Step 5: retrieve
def retrieve(question, vectors, chunks, k=TOP_K):
    q = get_embedder().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


# Step 6: generate
def build_prompt(question, retrieved):
    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_id']}] {c['text']}"
        for c, _ in retrieved
    )
    return (
        "Answer the question using only the context below.\n"
        "If the answer is not in the context, say \"I don't know based on the "
        "provided documents.\" Do not use outside knowledge.\n"
        "Mention which source (e.g. [Source: name, chunk N]) each part of your "
        "answer comes from.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def ask_llm(prompt):
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def ask(question, vectors, chunks, show=True):
    retrieved = retrieve(question, vectors, chunks)
    if show:
        print("\n=== RETRIEVED CHUNKS ===")
        for c, score in retrieved:
            print(f"\n[{c['source']}, chunk {c['chunk_id']}] (score {score:.3f})")
            print(c["text"][:300] + "...")
    answer = ask_llm(build_prompt(question, retrieved))
    if show:
        print("\n=== ANSWER ===")
        print(answer)
    return retrieved, answer


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        vectors, chunks = build_index()
    else:
        vectors, chunks = load_index()
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        ask(q, vectors, chunks)
