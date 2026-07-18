import pytest
import numpy as np
from rag.config import RAGConfig
from rag.ingestion.semantic_chunker import SemanticChunker
from rag.ingestion.parsers.base import ParsedPage

from rag.models.embedder import Embedder

class MockEmbedder(Embedder):
    def encode_batch(self, texts, prefix=""):
        # Return dummy vectors. We'll make the first 5 vectors close to each other,
        # and the next 5 vectors close to each other, but orthogonal across groups.
        vecs = []
        for i, text in enumerate(texts):
            v = np.zeros(10)
            if "Eiffel" in text:
                v[0] = 1.0  # Topic A
            else:
                v[1] = 1.0  # Topic B
            vecs.append(v)
        return vecs

def test_semantic_chunker_splits_topic_shifts(monkeypatch):
    config = RAGConfig()
    config.chunking.semantic_threshold = 0.5

    # Mock the embedder fetch
    import rag.models.model_manager
    class MockModelManager:
        def get_embedder(self, config):
            return MockEmbedder()
    
    monkeypatch.setattr(rag.models.model_manager, "get_model_manager", lambda: MockModelManager())

    chunker = SemanticChunker(config, MockEmbedder())
    
    text = (
        "The Eiffel Tower is located in Paris, France. It was built in 1889. "
        "It stands 330 metres tall. " * 2 +
        "Python is a programming language. It supports multiple paradigms. "
        "Python was created by Guido van Rossum. " * 2
    )
    page = ParsedPage(text=text, page=1)
    
    chunks = chunker.chunk_pages([page], "/fake.md", "fake.md", "md")
    
    # We should get at least 2 chunks because there is a clear semantic shift
    assert len(chunks) >= 2
    assert "Eiffel" in chunks[0].text
    assert "Python" in chunks[-1].text

def test_semantic_chunker_falls_back_for_short_text():
    config = RAGConfig()
    chunker = SemanticChunker(config, MockEmbedder())
    short = ParsedPage(text="One sentence only.")
    chunks = chunker.chunk_pages([short], "/f.md", "f.md", "md")
    assert len(chunks) == 1
