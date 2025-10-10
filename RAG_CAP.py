"""RAG demo for Capgemini PDF (Spanish)."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import json
from pathlib import Path

try:
    import faiss
except Exception:
    faiss = None


ROOT = Path(__file__).parent
PDF_PATH = ROOT / "Capgemini-CBE_2021_Spanish-v3.2.pdf"


def load_pdf_text(pdf_path: Path) -> str:
    """Load text from PDF using langchain's PyPDFLoader.

    Returns concatenated text of all pages.
    """
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    texts = [d.page_content for d in docs]
    return "\n\n".join(texts)


def build_cap_rag(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name: str = "google/flan-t5-small",
    llm_task: str = "text2text-generation",
    max_length: int = 256,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    k: int = 3,
    persist_dir: Path | str = None,
):
    """Load the CAP PDF, split into chunks, build embeddings and a FAISS index.

    Exposes `chunks_CAP` (list[str]) and `embeddings_CAP` (embedding model instance or similar)
    and returns a tuple (qa_chain, generator_only_callable, faiss_index).
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    # Load PDF text
    full_text = load_pdf_text(PDF_PATH)

    # Split into chunks and expose as `chunks_CAP`
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([full_text])
    chunks_CAP: List[str] = [d.page_content for d in docs]

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    embeddings_CAP = embedding_model  # keep the model instance exported under this name

    # Build FAISS vectorstore
    faiss_index = FAISS.from_texts(chunks_CAP, embedding_model)

    # Persistence: save chunks and index to disk if persist_dir provided
    if persist_dir:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        # save chunks
        with open(persist_path / "chunks_CAP.json", "w", encoding="utf-8") as f:
            json.dump(chunks_CAP, f, ensure_ascii=False, indent=2)
        # save FAISS index
        # Try langchain wrapper save
        saved = False
        try:
            faiss_index.save_local(str(persist_path / "faiss_index"))
            saved = True
        except Exception:
            try:
                faiss_index.persist(str(persist_path / "faiss_index"))
                saved = True
            except Exception:
                saved = False

        # If underlying faiss is available, try to write the raw faiss index
        if not saved and faiss is not None:
            try:
                # FAISS wrapper may expose a .index attribute
                raw_index = getattr(faiss_index, "index", None) or getattr(faiss_index, "faiss_index", None)
                if raw_index is not None:
                    faiss.write_index(raw_index, str(persist_path / "faiss_index.ivf"))
                    saved = True
            except Exception:
                saved = False

    # LLM (transformers pipeline) - expose the raw pipeline so we can call it directly
    llm_pipeline = pipeline(llm_task, model=llm_model_name, max_length=max_length)

    # Tokenizer for accurate truncation
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
    except Exception:
        # fallback: basic tokenizer settings
        tokenizer = None

    # RetrievalQA chain (kept for compatibility)
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm_pipeline),
        retriever=faiss_index.as_retriever(search_kwargs={"k": k}),
    )

    # Export variables and return
    return {
        "chunks_CAP": chunks_CAP,
        "embeddings_CAP": embeddings_CAP,
        "faiss_index": faiss_index,
        "qa_chain": qa_chain,
        "llm_pipeline": llm_pipeline,
        "tokenizer": tokenizer,
        "k": k,
    }


def compare_queries(
    faiss_index,
    llm_pipeline,
    queries: List[str],
    tokenizer=None,
    k: int = 3,
    context_token_limit: int = 256,
    answer_token_margin: int = 64,
):
    """For each query, retrieve top-k chunks from faiss_index, build a tokenizer-truncated context,
    then call llm_pipeline with and without context to compare RAG vs No-RAG.

    Returns a list of dicts with keys: query, rag, no_rag
    """
    results = []
    for q in queries:
        # Retrieve top-k docs
        try:
            docs = faiss_index.similarity_search(q, k=k)
        except Exception:
            # Fallback to retriever if vectorstore API differs
            retriever = faiss_index.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(q)

        # Build initial context from retrieved docs
        doc_texts = [getattr(d, "page_content", str(d)) for d in docs]

        # Determine model token budget
        model_max = None
        if tokenizer and hasattr(tokenizer, "model_max_length"):
            model_max = int(tokenizer.model_max_length)
            print(f"Model max tokens from tokenizer: {model_max}")
        # Fallback to provided context_token_limit
        if not model_max:
            model_max = 512

        # Prepare Spanish system instruction
        system_instruction = (
            "Eres un asistente experto en resumir y extraer información técnica de documentos empresariales. "
            "Responde siempre en español de forma breve, precisa y con referencia al contexto cuando esté disponible."
        )

        # Iteratively build context and ensure prompt fits into token budget
        current_text = "\n\n".join(doc_texts)

        def prompt_for(context_text, question_text):
            return (
                f"{system_instruction}\n\nContexto:\n{context_text}\n\nPregunta:\n{question_text}\n\nRespuesta:"
            )

        # Start with all retrieved docs, then iteratively truncate by removing last doc or truncating
        prompt = prompt_for(current_text, q)
        if tokenizer:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            token_len = tokens["input_ids"].shape[1]
        else:
            token_len = len(prompt) // 4

        allowed = model_max - answer_token_margin
        # If token_len is too big, first try dropping the last doc one by one
        docs_copy = doc_texts.copy()
        while token_len > allowed and len(docs_copy) > 0:
            docs_copy.pop()  # remove least relevant (assumed last)
            current_text = "\n\n".join(docs_copy)
            prompt = prompt_for(current_text, q)
            if tokenizer:
                tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
                token_len = tokens["input_ids"].shape[1]
            else:
                token_len = len(prompt) // 4

        # If still too long, truncate the context tokens directly to allowed size
        if token_len > allowed and tokenizer:
            # compute how many context tokens we can keep
            try:
                question_tokens = tokenizer(q, return_tensors="pt")["input_ids"].shape[1]
            except Exception:
                question_tokens = 32
            # keep space for instruction + question + margin
            keep_tokens = max(20, allowed - question_tokens - 20)
            # tokenize current_text and decode first keep_tokens
            ctokens = tokenizer(current_text, return_tensors="pt")["input_ids"][0]
            if len(ctokens) > keep_tokens:
                truncated_context = tokenizer.decode(ctokens[:keep_tokens], skip_special_tokens=True)
                current_text = truncated_context + "..."
            prompt = prompt_for(current_text, q)

        # Final prompts
        rag_prompt = prompt
        no_rag_prompt = f"{system_instruction}\n\nPregunta:\n{q}\n\nRespuesta (sin contexto):"

        # Call the transformers pipeline directly. Many pipelines accept strings and return list of dicts
        rag_out = llm_pipeline(rag_prompt)
        no_rag_out = llm_pipeline(no_rag_prompt)

        # Extract text depending on pipeline output format
        def _extract_text(out):
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                for key in ("generated_text", "text", "summary_text"):
                    if isinstance(first, dict) and key in first:
                        return first[key]
                if isinstance(first, str):
                    return first
            return str(out)

        rag_answer = _extract_text(rag_out)
        no_rag_answer = _extract_text(no_rag_out)

        results.append({"query": q, "rag": rag_answer, "no_rag": no_rag_answer})

    return results


if __name__ == "__main__":
    # Build artifacts
    artifacts = build_cap_rag()
    chunks_CAP = artifacts["chunks_CAP"]
    embeddings_CAP = artifacts["embeddings_CAP"]
    faiss_index = artifacts["faiss_index"]
    qa_chain = artifacts["qa_chain"]
    llm_pipeline = artifacts["llm_pipeline"]
    tokenizer = artifacts.get("tokenizer")

    print(f"Loaded {len(chunks_CAP)} chunks from {PDF_PATH.name}")

    # Example queries (replace with domain-relevant Spanish queries)
    sample_queries = [
        "¿Cómo se llama el documento?",
        "¿Qué significa trabajar con nuestros clientes?",
        "¿Cómo se debe manejar la información empresarial y financiera?",
        "¿Qué significa Tolerancia Cero?",
        "¿Hay resultados cuantitativos presentados?",
    ]

    comparisons = compare_queries(
        faiss_index,
        llm_pipeline,
        sample_queries,
        tokenizer=tokenizer,
        k=artifacts.get("k", 3),
        context_token_limit=800,
    )
    # Save comparisons to json
    out_path = ROOT / "rag_vs_no_rag_results.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, ensure_ascii=False, indent=2)
        print(f"Saved comparison results to {out_path}")
    except Exception as e:
        print(f"Failed to save comparison results: {e}")
    for c in comparisons:
        print("\n---\nQuery:", c["query"])
        print("RAG:\n", c["rag"])
        print("No-RAG:\n", c["no_rag"]) 