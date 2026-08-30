"""Tests for sequence detection and message classification."""

from ctxpact.compaction.detector import SequenceDetector
from ctxpact.config import PreserveConfig


def _msgs(roles: list[str]) -> list[dict]:
    return [{"role": r, "content": f"msg {i}"} for i, r in enumerate(roles)]


class TestSequenceDetector:
    def test_short_conversations_fully_retained(self):
        detector = SequenceDetector(retention_window=6)
        messages = _msgs(["system", "user", "assistant"])
        result = detector.classify(messages)
        assert len(result.retention_window) == 3
        assert len(result.compactible_messages) == 0

    def test_system_messages_always_preserved(self):
        detector = SequenceDetector(retention_window=2)
        messages = _msgs(["system", "user", "assistant", "user", "assistant",
                          "user", "assistant", "user", "assistant"])
        result = detector.classify(messages)
        assert len(result.system_messages) == 1
        assert result.system_messages[0]["role"] == "system"

    def test_retention_window_preserved(self):
        detector = SequenceDetector(retention_window=4)
        messages = _msgs(["system"] + ["user", "assistant"] * 6)
        result = detector.classify(messages)
        # The conservative merge may retain more than N, but the last N
        # messages must always be inside the retained region.
        assert len(result.retention_window) >= 4
        assert result.retention_window[-4:] == messages[-4:]

    def test_split_never_leaves_orphaned_tool_result_at_retained_start(self):
        # 9 non-system msgs, split_from_eviction = int(9 * 0.30) = 2 →
        # the raw split lands on a tool result. The retained region must
        # start at the assistant that issued that call.
        def tc(cid):
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": cid, "type": "function",
                                   "function": {"name": "x", "arguments": "{}"}}]}
        def tr(cid):
            return {"role": "tool", "content": "r", "tool_call_id": cid}

        messages = [
            {"role": "user", "content": "1"}, tc("c1"), tr("c1"),
            {"role": "user", "content": "2"}, tc("c2"), tr("c2"),
            {"role": "user", "content": "3"}, tc("c3"), tr("c3"),
        ]
        detector = SequenceDetector(retention_window=2)
        result = detector.classify(messages)
        assert result.retention_window[0]["role"] == "assistant"

        # The retained region must be a valid OpenAI pairing sequence:
        # every tool result follows the assistant that issued its call.
        pending = set()
        for m in result.retention_window:
            if m["role"] == "assistant":
                pending = {t["id"] for t in m.get("tool_calls") or []}
            elif m["role"] == "tool":
                assert m["tool_call_id"] in pending, "orphaned tool result"
                pending.discard(m["tool_call_id"])

    def test_user_messages_extracted_from_compactible(self):
        detector = SequenceDetector(
            retention_window=2,
            preserve_config=PreserveConfig(user_messages=True),
        )
        messages = _msgs(["system", "user", "assistant", "tool",
                          "assistant", "user", "assistant", "user", "assistant"])
        result = detector.classify(messages)
        # User messages in compactible region should be in preserved_user_messages
        assert all(m["role"] == "user" for m in result.preserved_user_messages)

    def test_conservative_merge_preserves_more(self):
        detector_conservative = SequenceDetector(
            retention_window=4,
            eviction_window=0.5,
            merge_strategy="conservative",
        )
        detector_aggressive = SequenceDetector(
            retention_window=4,
            eviction_window=0.5,
            merge_strategy="aggressive",
        )
        messages = _msgs(["user", "assistant"] * 10)

        result_c = detector_conservative.classify(messages)
        result_a = detector_aggressive.classify(messages)

        # Conservative should preserve at least as many as aggressive
        assert result_c.preserved_count >= result_a.preserved_count
