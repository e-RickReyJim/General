# q_and_a_setup.py

from datasets import load_dataset
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter


def build_qa_chain(
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.en",
    split: str = "train[:100]",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name: str = "google/flan-t5-small",
    llm_task: str = "text2text-generation",
    max_length: int = 256,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    k: int = 3,
) -> RetrievalQA:
    """Builds a simple RetrievalQA chain using local models and FAISS.

    Parameters are provided so the function can be tested easily with mocks.

    Returns:
        RetrievalQA: a configured RetrievalQA chain instance.
    """

    # Load dataset (small subset by default)
    wiki = load_dataset(dataset_name, dataset_config, split=split)
    texts = [x["text"] for x in wiki]

    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts)
    doc_texts = [doc.page_content for doc in docs]

    # Vector store
    db = FAISS.from_texts(doc_texts, embedding_model)

    # Local LLM
    qa_model = pipeline(llm_task, model=llm_model_name, max_length=max_length)
    llm = HuggingFacePipeline(pipeline=qa_model)

    # RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k})
    )

    return qa