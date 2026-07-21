from rag.config import RAGConfig
from rag.retrieval.expander import should_use_hyde


def test_should_use_hyde_false_for_t1():
    config = RAGConfig()
    config.resolved_tier = "T1"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("explain the methodology used in this paper", config) is False

def test_should_use_hyde_false_for_short_query():
    config = RAGConfig()
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("what is X", config) is False

def test_should_use_hyde_false_for_factual_prefix():
    config = RAGConfig()
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("who is the author of this document", config) is False

def test_should_use_hyde_true_for_complex():
    config = RAGConfig()
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "hyde"
    assert should_use_hyde("explain the relationship between the retrieval methods and accuracy", config) is True

def test_should_use_hyde_false_when_disabled():
    config = RAGConfig()
    config.resolved_tier = "T2"
    config.retrieval.query_expansion = "none"
    assert should_use_hyde("explain the relationship between retrieval and accuracy", config) is False
