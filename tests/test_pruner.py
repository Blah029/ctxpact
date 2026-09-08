"""Tests for Dynamic Context Pruning (Stage 1)."""

import json

from ctxpact.compaction.pruner import DynamicContextPruner
from ctxpact.config import DcpConfig


def _tool_call_msg(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool_result_msg(content: str, call_id: str = "call_test") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id}


class TestDeduplicateToolCalls:
    def test_removes_duplicate_tool_calls(self):
        pruner = DynamicContextPruner(DcpConfig(dedup_tool_calls=True))
        messages = [
            {"role": "user", "content": "search for X"},
            _tool_call_msg("search", {"query": "X"}),
            _tool_result_msg("result 1"),
            _tool_call_msg("search", {"query": "X"}),  # Duplicate
            _tool_result_msg("result 2"),
        ]
        result = pruner.prune(messages)
        # Should remove the first duplicate, keep the last
        assert result.deduped_tool_calls >= 1
        assert len(result.messages) < len(messages)

    def test_keeps_different_tool_calls(self):
        pruner = DynamicContextPruner(DcpConfig(dedup_tool_calls=True))
        messages = [
            _tool_call_msg("search", {"query": "X"}),
            _tool_result_msg("result 1"),
            _tool_call_msg("search", {"query": "Y"}),  # Different args
            _tool_result_msg("result 2"),
        ]
        result = pruner.prune(messages)
        assert result.deduped_tool_calls == 0
        assert len(result.messages) == len(messages)


class TestStripSupersededWrites:
    def test_keeps_only_latest_write(self):
        pruner = DynamicContextPruner(DcpConfig(strip_superseded_writes=True))
        messages = [
            _tool_call_msg("write_file", {"path": "main.py", "content": "v1"}),
            _tool_result_msg("written"),
            _tool_call_msg("write_file", {"path": "main.py", "content": "v2"}),
            _tool_result_msg("written"),
            _tool_call_msg("write_file", {"path": "main.py", "content": "v3"}),
            _tool_result_msg("written"),
        ]
        result = pruner.prune(messages)
        assert result.superseded_writes >= 2  # Removed 2 earlier writes + results


class TestTruncateErrors:
    def test_truncates_long_tracebacks(self):
        pruner = DynamicContextPruner(DcpConfig(truncate_errors=True))
        long_trace = "Traceback (most recent call last):\n" + "\n".join(
            [f"  File line {i}" for i in range(50)]
        ) + "\nValueError: something broke"

        messages = [{"role": "tool", "content": long_trace}]
        result = pruner.prune(messages)
        assert result.truncated_errors == 1
        content = result.messages[0]["content"]
        assert "truncated by ctxpact" in content
        assert len(content) < len(long_trace)


class TestStripToolPayloads:
    def test_strips_verbose_tool_results(self):
        pruner = DynamicContextPruner(DcpConfig(strip_tool_payloads=True))
        verbose_output = "x" * 1000
        messages = [{"role": "tool", "content": verbose_output}]
        result = pruner.prune(messages)
        assert result.stripped_payloads == 1
        assert len(result.messages[0]["content"]) < 500

    def test_preserves_short_tool_results(self):
        pruner = DynamicContextPruner(DcpConfig(strip_tool_payloads=True))
        messages = [{"role": "tool", "content": "OK"}]
        result = pruner.prune(messages)
        assert result.stripped_payloads == 0
        assert result.messages[0]["content"] == "OK"


# ---------------------------------------------------------------------------
# Retention-window protection (commit d52e6ee)
# ---------------------------------------------------------------------------

WINDOW = 6  # matches config.yaml stage2_summarize.retention_window


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def _tr(content: str, call_id: str) -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def _assistant_with(ids: list[tuple[str, dict, str]]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for name, args, cid in ids
        ],
    }


def _assert_valid_tool_pairing(messages: list[dict]) -> None:
    """Every tool message must follow an assistant message whose tool_calls
    include the matching id (OpenAI sequencing requirement)."""
    pending: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            pending = {tc["id"] for tc in msg.get("tool_calls") or []}
        elif msg.get("role") == "tool":
            assert msg.get("tool_call_id") in pending, (
                f"orphaned tool result: {msg.get('tool_call_id')}"
            )
            pending.discard(msg["tool_call_id"])


class TestRetentionPolicy:
    def test_recent_tool_result_preserved_when_history_longer_than_window(self):
        big = "x" * 1200  # > 500 chars, strip candidate
        messages = [{"role": "user", "content": "start"}]
        for k in range(10):
            messages.append(_tc("read_file", {"path": f"f{k}.py"}, f"c{k}"))
            messages.append(_tr(big, f"c{k}"))
        messages.append({"role": "assistant", "content": "done"})

        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        tools = [m for m in result.messages if m["role"] == "tool"]
        # Most recent tool results are inside the protected tail
        assert tools[-1]["content"] == big
        assert tools[-2]["content"] == big
        # Older payloads were stripped
        assert result.stripped_payloads > 0
        _assert_valid_tool_pairing(result.messages)

    def test_recent_traceback_preserved_within_window(self):
        traceback = (
            "Traceback (most recent call last):\n"
            + "\n".join(f"  File l{i}" for i in range(40))
            + "\nValueError: boom"
        )
        messages = [{"role": "user", "content": "run it"}]
        for k in range(3):
            messages.append(_tc("bash", {"cmd": f"cmd{k}"}, f"c{k}"))
            messages.append(_tr(traceback, f"c{k}"))

        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        # 7 messages, window 6 — only the first (user) msg is mutable
        assert result.truncated_errors == 0
        assert result.messages[-1]["content"] == traceback

    def test_history_exactly_window_length_fully_protected(self):
        big = "y" * 4000
        messages = [
            {"role": "user", "content": "read big.py"},
            _tc("read_file", {"path": "big.py"}, "c1"),
            _tr(big, "c1"),
            {"role": "assistant", "content": "ok"},
            _tc("bash", {"cmd": "ls"}, "c2"),
            _tr(big, "c2"),
        ]
        assert len(messages) == WINDOW

        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        assert result.stripped_payloads == 0
        assert result.messages[-1]["content"] == big

    def test_history_shorter_than_window_fully_protected(self):
        big = "z" * 4000
        messages = [
            {"role": "user", "content": "ls"},
            _tc("bash", {"cmd": "ls"}, "c1"),
            _tr(big, "c1"),
        ]

        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        # The whole history is inside the window — nothing may be stripped
        assert result.stripped_payloads == 0
        assert result.messages[-1]["content"] == big

    def test_legacy_behavior_without_retention_window(self):
        big = "w" * 4000
        messages = [
            {"role": "user", "content": "ls"},
            _tc("bash", {"cmd": "ls"}, "c1"),
            _tr(big, "c1"),
        ]
        pruner = DynamicContextPruner(DcpConfig())  # retention_window=0
        result = pruner.prune(messages)
        assert result.stripped_payloads == 1

    def test_dedup_keeps_latest_occurrence_within_window(self):
        messages = [
            {"role": "user", "content": "search"},
            _tc("search", {"q": "X"}, "c1"),
            _tr("r1", "c1"),
            _tc("search", {"q": "X"}, "c2"),
            _tr("r2", "c2"),
            {"role": "assistant", "content": "found it"},
        ]
        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)
        tools = [m for m in result.messages if m["role"] == "tool"]
        assert tools[-1]["content"] == "r2"
        _assert_valid_tool_pairing(result.messages)

    def test_parallel_call_dedup_produces_no_orphaned_tool_results(self):
        messages = [
            {"role": "user", "content": "do both"},
            _assistant_with([
                ("grep", {"q": "A"}, "ca"),
                ("ls", {}, "cb"),
            ]),
            _tr("A result", "ca"),
            _tr("B result", "cb"),
            _assistant_with([("grep", {"q": "A"}, "cc")]),
            _tr("A result 2", "cc"),
        ]
        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        # The duplicate assistant batch is removed together with ALL of its
        # results — "B result" must not survive orphaned.
        assert result.deduped_tool_calls > 0
        assert all(m["content"] != "B result" for m in result.messages)
        _assert_valid_tool_pairing(result.messages)

    def test_superseded_write_keeps_latest_and_no_orphans(self):
        messages = [
            {"role": "user", "content": "write it"},
            _tc("write_file", {"path": "a.py"}, "c1"),
            _tr("wrote v1", "c1"),
            _tc("write_file", {"path": "a.py"}, "c2"),
            _tr("wrote v2", "c2"),
        ]
        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        tools = [m for m in result.messages if m["role"] == "tool"]
        assert tools[-1]["content"] == "wrote v2"
        _assert_valid_tool_pairing(result.messages)

    def test_parallel_superseded_write_produces_no_orphans(self):
        messages = [
            {"role": "user", "content": "write and read"},
            _assistant_with([
                ("write_file", {"path": "a.py"}, "ca"),
                ("read_file", {"path": "b.py"}, "cb"),
            ]),
            _tr("wrote v1", "ca"),
            _tr("b.py contents", "cb"),
            _tc("write_file", {"path": "a.py"}, "cc"),
            _tr("wrote v2", "cc"),
        ]
        pruner = DynamicContextPruner(DcpConfig(), retention_window=WINDOW)
        result = pruner.prune(messages)

        # The first batch is removed wholesale; "b.py contents" (belonging to
        # the removed assistant) must not survive orphaned.
        assert all(m["content"] != "b.py contents" for m in result.messages)
        assert any(m["content"] == "wrote v2" for m in result.messages)
        _assert_valid_tool_pairing(result.messages)
