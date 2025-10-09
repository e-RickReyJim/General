import pytest
from unittest.mock import patch, MagicMock

from q_and_a_setup import build_qa_chain


@patch("q_and_a_setup.load_dataset")
@patch("q_and_a_setup.HuggingFaceEmbeddings")
@patch("q_and_a_setup.FAISS")
@patch("q_and_a_setup.pipeline")
@patch("q_and_a_setup.HuggingFacePipeline")
@patch("q_and_a_setup.RetrievalQA")
def test_build_qa_chain_mocks(
    mock_retrieval_qa, mock_hf_pipeline, mock_pipeline, mock_faiss, mock_embeddings, mock_load_dataset
):
    # Mock dataset
    mock_load_dataset.return_value = [
        {"text": "Document 1 content."},
        {"text": "Document 2 content."},
    ]

    # Mock embeddings instance
    mock_embeddings.return_value = MagicMock(name="embeddings")

    # Mock FAISS.from_texts to return a fake db with as_retriever
    fake_db = MagicMock()
    fake_db.as_retriever.return_value = MagicMock(name="retriever")
    mock_faiss.from_texts.return_value = fake_db

    # Mock pipeline and HuggingFacePipeline
    fake_pipeline = MagicMock(name="pipeline")
    mock_pipeline.return_value = fake_pipeline
    mock_hf_pipeline.return_value = MagicMock(name="hf_pipeline")

    # Mock RetrievalQA.from_chain_type to return a sentinel
    qa_sentinel = MagicMock(name="qa")
    mock_retrieval_qa.from_chain_type.return_value = qa_sentinel

    # Run
    qa = build_qa_chain(
        dataset_name="some/dataset",
        dataset_config="cfg",
        split="train[:2]",
        embedding_model_name="embed-model",
        llm_model_name="llm-model",
        llm_task="text2text-generation",
        max_length=16,
        chunk_size=10,
        chunk_overlap=0,
        k=2,
    )

    # Validate returned object is the sentinel from RetrievalQA.from_chain_type
    assert qa is qa_sentinel
    mock_load_dataset.assert_called_once_with("some/dataset", "cfg", split="train[:2]")
    mock_embeddings.assert_called_once_with(model_name="embed-model")
    mock_faiss.from_texts.assert_called()
    mock_pipeline.assert_called_once_with("text2text-generation", model="llm-model", max_length=16)
    mock_hf_pipeline.assert_called_once()
