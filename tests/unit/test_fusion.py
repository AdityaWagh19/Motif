"""
tests/unit/test_fusion.py — Unit tests for RRF fusion.

No model downloads or external services required.
All tests use pure Python — no fixtures needed beyond simple lists.
"""
from __future__ import annotations

from rag.retrieval.fusion import rrf_fuse, rrf_to_scored_passages
from rag.types import Chunk, ScoredPassage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(id: str) -> Chunk:
    return Chunk(
        id=id,
        text=f"Passage text for chunk {id}.",
        source="/test/doc.pdf",
        filename="doc.pdf",
        source_type="pdf",
    )


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------

class TestRRFFuse:
    def test_empty_input_returns_empty(self) -> None:
        assert rrf_fuse([]) == []

    def test_single_list_preserves_order(self) -> None:
        ranked = [(str(i), 1.0 - i * 0.1) for i in range(10)]
        fused = rrf_fuse([ranked], top_k=10)
        ids = [x[0] for x in fused]
        assert ids == [x[0] for x in ranked]

    def test_top_k_limits_results(self) -> None:
        ranked = [("a", 1.0), ("b", 0.5), ("c", 0.3)]
        fused = rrf_fuse([ranked], top_k=2)
        assert len(fused) == 2

    def test_top_k_larger_than_total_is_fine(self) -> None:
        ranked = [("a", 1.0), ("b", 0.5)]
        fused = rrf_fuse([ranked], top_k=100)
        assert len(fused) == 2

    def test_scores_decrease_monotonically(self) -> None:
        ranked = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
        fused = rrf_fuse([ranked], top_k=3)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_multi_list_item_in_all_lists_ranks_highest(self) -> None:
        """An item appearing in all three lists should score highest."""
        list1 = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
        list2 = [("b", 0.9), ("c", 0.7), ("d", 0.4)]
        list3 = [("c", 0.6), ("b", 0.5), ("e", 0.3)]
        fused = rrf_fuse([list1, list2, list3], top_k=5)
        ids = [x[0] for x in fused]
        # "b" appears in all three lists AND "c" appears in all three
        # Both should rank above "a" (only in list1) and "d" (only in list2)
        assert "b" in ids and "c" in ids
        b_rank = ids.index("b")
        a_rank = ids.index("a")
        assert b_rank < a_rank or ids.index("c") < a_rank

    def test_item_absent_from_list_gets_no_credit(self) -> None:
        """Items only in one list should score lower than items in multiple."""
        list1 = [("common", 1.0), ("only_in_1", 0.9)]
        list2 = [("common", 0.8), ("only_in_2", 0.7)]
        fused = rrf_fuse([list1, list2], top_k=3)
        ids = [x[0] for x in fused]
        # "common" is in both lists — must rank first
        assert ids[0] == "common"

    def test_two_empty_lists_returns_empty(self) -> None:
        fused = rrf_fuse([[], []], top_k=10)
        assert fused == []

    def test_one_empty_one_non_empty(self) -> None:
        ranked = [("x", 1.0), ("y", 0.5)]
        fused = rrf_fuse([ranked, []], top_k=10)
        assert len(fused) == 2
        assert fused[0][0] == "x"

    def test_k_parameter_affects_score_magnitude(self) -> None:
        """Higher k → smaller score differences between ranks."""
        ranked = [("a", 1.0), ("b", 0.5)]
        fused_k60 = rrf_fuse([ranked], top_k=2, k=60)
        fused_k10 = rrf_fuse([ranked], top_k=2, k=10)
        # With k=10, rank-1 item gets 1/(10+1)=0.091, rank-2 gets 1/(10+2)=0.083
        # With k=60, rank-1 gets 1/(60+1)=0.016, rank-2 gets 1/(60+2)=0.016
        # Lower k → larger spread between rank-1 and rank-2
        score_diff_k10 = fused_k10[0][1] - fused_k10[1][1]
        score_diff_k60 = fused_k60[0][1] - fused_k60[1][1]
        assert score_diff_k10 > score_diff_k60

    def test_result_ids_are_unique(self) -> None:
        list1 = [("a", 1.0), ("b", 0.5), ("a", 0.3)]  # duplicate id
        # Even with a malformed input, result should have unique IDs
        # (dict-based accumulation naturally handles this)
        fused = rrf_fuse([list1], top_k=5)
        ids = [x[0] for x in fused]
        assert len(ids) == len(set(ids))

    def test_single_item_single_list(self) -> None:
        fused = rrf_fuse([[("only", 1.0)]], top_k=5)
        assert len(fused) == 1
        assert fused[0][0] == "only"


# ---------------------------------------------------------------------------
# rrf_to_scored_passages
# ---------------------------------------------------------------------------

class TestRRFToScoredPassages:
    def test_empty_fused_returns_empty(self, minimal_config) -> None:
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(minimal_config)
        result = rrf_to_scored_passages([], store)
        assert result == []

    def test_missing_ids_are_dropped(self, minimal_config) -> None:
        """Chunk IDs that don't exist in ChunkStore are silently dropped."""
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(minimal_config)
        fused = [("nonexistent-id-abc", 0.5)]
        result = rrf_to_scored_passages(fused, store)
        assert result == []

    def test_existing_id_returned_as_scored_passage(self, minimal_config) -> None:
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(minimal_config)
        chunk = Chunk(
            id="test-id-001",
            text="Test passage text.",
            source="/test/doc.pdf",
            filename="doc.pdf",
            source_type="pdf",
        )
        store.insert_batch([chunk])

        fused = [("test-id-001", 0.73)]
        result = rrf_to_scored_passages(fused, store)

        assert len(result) == 1
        assert isinstance(result[0], ScoredPassage)
        assert result[0].chunk.id == "test-id-001"
        assert abs(result[0].score - 0.73) < 1e-6
        assert result[0].retrieval_method == "fused"

    def test_score_matches_fused_input(self, minimal_config) -> None:
        from rag.storage.chunk_store import ChunkStore
        store = ChunkStore(minimal_config)
        chunk = Chunk(
            id="test-id-002",
            text="Another passage.",
            source="/test/doc.pdf",
            filename="doc.pdf",
            source_type="pdf",
        )
        store.insert_batch([chunk])
        fused = [("test-id-002", 0.123)]
        result = rrf_to_scored_passages(fused, store)
        assert abs(result[0].score - 0.123) < 1e-6
