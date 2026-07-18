import pytest
from pathlib import Path
from rag.ingestion import ingest_path, sync_directory
from rag.storage.chunk_store import ChunkStore
from rag.config import RAGConfig

@pytest.fixture
def minimal_config(tmp_path):
    config = RAGConfig()
    config.storage.db_path = str(tmp_path / ".ragdb")
    config.resolved_tier = "T1"
    config.chunking.use_semantic = False
    return config

@pytest.fixture(autouse=True)
def skip_if_no_model(minimal_config):
    model_path = Path(minimal_config.models.embed_model)
    if not model_path.is_absolute():
        model_path = model_path.resolve()
    if not model_path.exists():
        pytest.skip(f"Embedding model not found at {model_path}.")

@pytest.mark.slow
def test_sync_detects_new_file(minimal_config, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    f1 = d / "doc1.md"
    f1.write_text("First document text " * 20)
    ingest_path(f1, config=minimal_config, recursive=False, console=None)

    # Add a new file
    f2 = d / "doc2.md"
    f2.write_text("Second document text " * 20)
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    
    assert result.added > 0
    assert ChunkStore(minimal_config).count_documents() == 2

@pytest.mark.slow
def test_sync_detects_deleted_file(minimal_config, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    f1 = d / "doc1.md"
    f1.write_text("Document to be deleted " * 20)
    f2 = d / "doc2.md"
    f2.write_text("Document that stays " * 20)
    
    ingest_path(d, config=minimal_config, recursive=False, console=None)
    assert ChunkStore(minimal_config).count_documents() == 2
    
    f1.unlink()
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    
    assert result.removed > 0
    assert ChunkStore(minimal_config).count_documents() == 1

@pytest.mark.slow
def test_sync_detects_changed_file(minimal_config, tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    f = d / "doc.md"
    f.write_text("Original content here " * 20)
    
    ingest_path(f, config=minimal_config, recursive=False, console=None)
    
    old_count = ChunkStore(minimal_config).count()
    f.write_text("Completely new different content about databases " * 30)
    
    result = sync_directory(d, config=minimal_config, recursive=False, console=None)
    
    assert result.reindexed > 0
