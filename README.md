# Minimal RAG App
Answers questions about 8 space-exploration Wikipedia articles using local embeddings (sentence-transformers) and a local LLM (Ollama, llama3.2).

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```
No API keys are needed. If you swap in a hosted LLM, set its key as an environment variable, never in code.

## Add documents
Run `python download_docs.py`, or put your own `.txt` files in `docs/`.
Then rebuild the index: `python rag.py --rebuild`.

## Run
- Interactive: `python rag.py`
- Test questions: `python run_tests.py` (writes `results_table.md`)

## Settings
Chunk size 150 words, overlap 30, k=4, embeddings all-MiniLM-L6-v2, LLM llama3.2.
