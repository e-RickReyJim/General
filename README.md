# q_and_a_setup

This small module provides a helper `build_qa_chain` that wires together dataset loading, embeddings, FAISS indexing, and a local LLM into a LangChain `RetrievalQA` chain.

Usage example

```python
from q_and_a_setup import build_qa_chain

# Default (will download models/datasets)
qa = build_qa_chain()

# Parameterized (faster to test by pointing to smaller splits or local models)
qa = build_qa_chain(split="train[:100]", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")

# qa is a RetrievalQA chain; call qa.run("Your question") or qa("Your question") depending on the chain interface
```

Testing

Run the unit tests (they mock the heavy dependencies):

```powershell
python -m pip install -r requirements.txt
pytest -q
```

# General

For learning, testing, notebooks.
