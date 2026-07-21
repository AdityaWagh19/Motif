"""
tests/unit/test_context_builder.py — Unit tests for context assembly.

No models required.
"""
from __future__ import annotations

from rag.generation.context_builder import ContextBuilder, _anti_middle_order
from rag.types import Chunk, ScoredPassage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passage(score: float, text: str, id: str = "id") -> ScoredPassage:
    return ScoredPassage(
        chunk=Chunk(
            id=id,
            text=text,
            source="/test.pdf",
            filename="test.pdf",
            source_type="pdf",
        ),
        score=score,
        retrieval_method="dense",
    )


# ---------------------------------------------------------------------------
# _anti_middle_order
# ---------------------------------------------------------------------------

class TestAntiMiddleOrder:
    def test_empty_list(self) -> None:
        assert _anti_middle_order([]) == []

    def test_single_element(self) -> None:
        p1 = _passage(1.0, "p1")
        assert _anti_middle_order([p1]) == [p1]

    def test_two_elements_preserves_order(self) -> None:
        p1 = _passage(1.0, "best")
        p2 = _passage(0.5, "second")
        assert _anti_middle_order([p1, p2]) == [p1, p2]

    def test_five_elements_ordered_correctly(self) -> None:
        p1 = _passage(5.0, "first", id="p1")
        p2 = _passage(4.0, "second", id="p2")
        p3 = _passage(3.0, "third", id="p3")
        p4 = _passage(2.0, "fourth", id="p4")
        p5 = _passage(1.0, "fifth", id="p5")

        passages = [p1, p2, p3, p4, p5]
        ordered = _anti_middle_order(passages)

        ids = [p.chunk.id for p in ordered]
        # Expected: Best at 0, 2nd best at -1, others fill middle
        assert ids == ["p1", "p3", "p4", "p5", "p2"]

    def test_handles_unsorted_input_by_sorting_first(self) -> None:
        p2 = _passage(4.0, "second", id="p2")
        p1 = _passage(5.0, "first", id="p1")
        p3 = _passage(3.0, "third", id="p3")

        # Input is scrambled
        ordered = _anti_middle_order([p2, p1, p3])

        # Best (p1) goes first, second (p2) goes last
        ids = [p.chunk.id for p in ordered]
        assert ids == ["p1", "p3", "p2"]


# ---------------------------------------------------------------------------
# ContextBuilder.build
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def test_empty_passages_returns_empty(self, minimal_config) -> None:
        builder = ContextBuilder()
        prompt, used = builder.build([], "query", [], minimal_config)
        assert prompt == ""
        assert used == []

    def test_budget_enforced_by_word_count(self, minimal_config) -> None:
        minimal_config.generation.context_max_tokens = 512
        # Budget words ≈ 512 * 0.75 = 384 words.
        # Overhead is 200 words. Available = 184 words.
        
        # Passage 1: 150 words (fits)
        p1 = _passage(1.0, "word " * 150, id="p1")
        # Passage 2: 50 words (exceeds budget: 150 + 50 = 200 > 184)
        p2 = _passage(0.5, "word " * 50, id="p2")

        builder = ContextBuilder()
        prompt, used = builder.build([p1, p2], "q", [], minimal_config)

        assert len(used) == 1
        assert used[0].chunk.id == "p1"

    def test_always_includes_one_passage_even_if_over_budget(self, minimal_config) -> None:
        minimal_config.generation.context_max_tokens = 100
        # Budget words ≈ 75. Overhead = 200. Available = -125 (less than zero).
        
        # Passage: 500 words
        p1 = _passage(1.0, "word " * 500, id="p1")

        builder = ContextBuilder()
        prompt, used = builder.build([p1], "q", [], minimal_config)

        assert len(used) == 1
        assert used[0].chunk.id == "p1"

    def test_history_consumes_budget(self, minimal_config) -> None:
        minimal_config.generation.context_max_tokens = 512
        # Budget ≈ 384. Overhead = 200. Base available = 184.

        # Add history with 100 words. Available should drop to 84.
        history = [
            {"role": "user", "content": "hello " * 50},
            {"role": "assistant", "content": "world " * 50},
        ]

        # Passage: 100 words.
        # Without history, this would fit (100 <= 184).
        # With history, it doesn't (100 > 84), BUT one passage is always included.
        # Let's use two passages to prove the second is dropped.
        p1 = _passage(1.0, "word " * 50, id="p1")
        p2 = _passage(0.9, "word " * 50, id="p2")

        builder = ContextBuilder()
        prompt, used = builder.build([p1, p2], "q", history, minimal_config)

        # 50 words fits in 84. Second 50 brings total to 100 > 84, so p2 dropped.
        assert len(used) == 1
        assert used[0].chunk.id == "p1"

    def test_history_included_in_prompt(self, minimal_config) -> None:
        history = [{"role": "user", "content": "Previous question"}]
        
        p1 = _passage(1.0, "Doc text")
        builder = ContextBuilder()
        prompt, _ = builder.build([p1], "Current question", history, minimal_config)

        assert "Previous question" in prompt
        assert "Current question" in prompt
        assert "Doc text" in prompt
        assert "Prior context:" in prompt  # from HISTORY_SYSTEM_PROMPT
