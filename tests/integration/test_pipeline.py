"""
tests/integration/test_pipeline.py
"""
from unittest.mock import MagicMock, patch

from rag.config import RAGConfig
from rag.pipeline import QueryPipeline
from rag.types import AnswerResult


def test_query_pipeline_glue_code(tmp_path):
    config = RAGConfig()
    config.storage.db_path = str(tmp_path)
    config.storage.query_cache_enabled = False
    
    # We patch the components used by QueryPipeline to avoid loading real models
    # and to verify that the glue code calls them correctly.
    with patch("rag.pipeline.get_model_manager") as mock_gmm, \
         patch("rag.intent.IntentClassifier") as mock_intent, \
         patch("rag.pipeline.QueryExpander") as mock_expander, \
         patch("rag.pipeline.VectorStore") as mock_vs, \
         patch("rag.pipeline.BM25Index") as mock_bm25, \
         patch("rag.pipeline.ChunkStore") as mock_cs, \
         patch("rag.pipeline.rerank") as mock_rerank:
         
        # Mock IntentClassifier to return normal question (not greeting/chitchat)
        from rag.intent import Intent
        mock_intent.return_value.classify.return_value = Intent.QUERY
        
        # Mock VectorStore, BM25, and ChunkStore returns
        mock_vs.return_value.search_dense.return_value = [("chunk_1", 0.9)]
        mock_bm25.return_value.search.return_value = []
        mock_cs.return_value.count.return_value = 1
        
        from rag.types import Chunk
        mock_cs.return_value.fetch_batch.return_value = [
            Chunk(id="chunk_1", text="dummy", source="f", filename="f", source_type="txt", token_count=1, indexed_at="")
        ]
        
        # Mock rerank to return the passage untouched
        mock_rerank.side_effect = lambda *args, **kwargs: args[1]
        
        # Mock QueryExpander
        mock_expander.return_value.expand.return_value = ([0.1] * 768, "Hi there!")
        
        # Mock ModelManager and LLMClient
        mock_mm = MagicMock()
        mock_gmm.return_value = mock_mm
        mock_llm = MagicMock()
        mock_mm.get_llm.return_value = mock_llm
        
        # We need to simulate generator returning a generator for LLM stream
        def fake_generate(*args, **kwargs):
            yield "Hello "
            yield "world!"
        mock_llm.stream.side_effect = fake_generate
        
        pipeline = QueryPipeline(config)
        
        # Test pipeline.answer
        result = pipeline.answer("Hi there!", history=[], show_sources=False)
        
        # Assertions
        assert isinstance(result, AnswerResult)
        assert result.text == "Hello world!"
        assert mock_mm.get_llm.called
