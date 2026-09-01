"""Tests for the stateless handoff decision (marker-based dedup)."""

from ctxpact.compaction.prompts import (
    FULL_HANDOFF_PROMPT,
    HANDOFF_MARKER,
    HANDOFF_REMINDER_PROMPT,
    REMINDER_MARKER,
)
from ctxpact.server import _handoff_decision


class TestHandoffDecision:
    def test_no_markers_yields_full(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        kind, msg = _handoff_decision(msgs)
        assert kind == "full"
        assert msg["role"] == "system"
        assert HANDOFF_MARKER in msg["content"]

    def test_full_marker_yields_reminder(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant",
             "content": f"{HANDOFF_MARKER}\ntask, decisions, paths, next steps"},
            {"role": "user", "content": "carry on"},
        ]
        kind, msg = _handoff_decision(msgs)
        assert kind == "reminder"
        assert REMINDER_MARKER in msg["content"]

    def test_both_markers_yields_none(self):
        msgs = [
            {"role": "assistant", "content": f"{HANDOFF_MARKER} ..."},
            {"role": "assistant", "content": f"{REMINDER_MARKER} ..."},
        ]
        kind, msg = _handoff_decision(msgs)
        assert kind is None
        assert msg is None

    def test_multipart_content_is_scanned(self):
        msgs = [
            {"role": "assistant",
             "content": [{"type": "text", "text": f"before {HANDOFF_MARKER} after"}]},
        ]
        kind, _ = _handoff_decision(msgs)
        assert kind == "reminder"

    def test_non_string_content_does_not_crash(self):
        msgs = [{"role": "assistant", "content": None}]
        kind, _ = _handoff_decision(msgs)
        assert kind == "full"


class TestPrompts:
    def test_full_prompt_contains_marker(self):
        assert HANDOFF_MARKER in FULL_HANDOFF_PROMPT

    def test_reminder_prompt_contains_marker(self):
        assert REMINDER_MARKER in HANDOFF_REMINDER_PROMPT

    def test_reminder_does_not_contain_full_marker(self):
        assert HANDOFF_MARKER not in HANDOFF_REMINDER_PROMPT
