"""
tests/integration/test_history.py — Session history persistence tests.
"""
from __future__ import annotations

from rag.session import Session


def test_session_history_persists_across_restart(minimal_config, tmp_db_root) -> None:
    session1 = Session(minimal_config)
    session1.add_turn("first query", "first answer")
    session1.save()

    session2 = Session(minimal_config)
    loaded = session2.load()
    assert loaded is True
    assert session2.turn_count == 1
    assert session2.last_query == "first query"

def test_session_clear_deletes_file(minimal_config, tmp_db_root) -> None:
    session = Session(minimal_config)
    session.add_turn("q", "a")
    session.save()
    assert session.history_path.exists()
    
    session.clear()
    assert not session.history_path.exists()
    assert session.turn_count == 0

def test_session_new_archives(minimal_config, tmp_db_root) -> None:
    session = Session(minimal_config)
    session.add_turn("q1", "a1")
    session.save()
    
    archive_path = session.new()
    assert archive_path is not None
    assert archive_path.exists()
    assert session.turn_count == 0
    # Original file is moved (or rather copied then cleared)
    assert not session.history_path.exists()

def test_empty_session_doesnt_crash(minimal_config, tmp_db_root) -> None:
    session = Session(minimal_config)
    # No save
    assert not session.load()
    assert session.turn_count == 0
    assert session.last_query is None
    
    session.save() # Saving empty list is fine
    assert session.history_path.exists()
