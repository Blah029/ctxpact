"""Summarization prompt templates for context compaction.

Inspired by Anthropic SDK's <summary> tag approach and Forge's
custom prompt support.
"""

COMPACTION_SYSTEM_PROMPT = """\
You are a context compaction assistant. Your job is to create a concise \
summary of conversation history that allows the conversation to continue \
seamlessly."""

COMPACTION_USER_PROMPT = """\
The conversation below needs to be compacted to fit within context limits. \
Create a structured summary that preserves essential information.

<conversation_to_compact>
{conversation}
</conversation_to_compact>

Create a summary that preserves:

1. **Task State**: What was being worked on, current progress, overall goal
2. **Key Decisions**: Choices made with rationale
3. **Important Data**: File paths, variable names, configuration values
4. **Tool Outcomes**: What tool calls were made and their results (not raw output)
5. **Errors & Fixes**: Error messages encountered and how they were resolved
6. **Open Items**: Unresolved questions, next steps, blockers

Rules:
- Preserve all file paths and line numbers exactly
- Keep error type/message verbatim (not full traces)
- Code blocks under {code_line_limit} lines: keep in full
- Code blocks over {code_line_limit} lines: keep signature + key modified sections
- Be concise but complete — prevent duplicate work on resume
- Write in a way that enables immediate continuation of the task

Wrap your summary in <summary></summary> tags."""


def build_compaction_prompt(
    conversation_text: str,
    code_line_limit: int = 50,
    custom_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the messages to send for summarization.

    Returns OpenAI-format messages ready to send to the summarization model.
    """
    if custom_prompt:
        return [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": custom_prompt.format(
                conversation=conversation_text,
                code_line_limit=code_line_limit,
            )},
        ]

    return [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPACTION_USER_PROMPT.format(
                conversation=conversation_text,
                code_line_limit=code_line_limit,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Handoff prompts — injected into the model right after compaction
# ---------------------------------------------------------------------------

# Visible markers: the model is instructed to emit these exact lines. They are
# scanned from conversation content (stateless) so the handoff fires once per
# chat even when the client sends no stable session id.
HANDOFF_MARKER = "[CTX-HANDOFF v1]"
REMINDER_MARKER = "[CTX-REMINDER v1]"

FULL_HANDOFF_PROMPT = f"""\
The gateway has just compacted this conversation's history to fit the model's \
context. In your reply, briefly warn the user, then provide a HANDOFF SUMMARY \
the user can paste as the first message of a new chat to continue: task & goal, \
key decisions, file paths / line numbers, open errors, next steps. Start the \
summary with the exact line: {HANDOFF_MARKER}"""

HANDOFF_REMINDER_PROMPT = f"""\
The gateway compacted this conversation a second time; most earlier history is \
now unavailable. Strongly advise the user to start a new chat using the earlier \
handoff summary. Do not repeat the summary. Start your reply with the exact \
line: {REMINDER_MARKER}"""


def format_messages_for_summary(messages: list[dict]) -> str:
    """Convert a list of messages into a readable text block for summarization."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        if isinstance(content, list):
            # Multi-part: extract text parts
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            content = "\n".join(text_parts)

        # Truncate very long messages for the summary input
        if isinstance(content, str) and len(content) > 2000:
            content = content[:1500] + "\n... [truncated for compaction] ..."

        lines.append(f"[{role}]: {content}")

        # Include tool call info
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            lines.append(f"  → Tool call: {fn.get('name', '?')}({fn.get('arguments', '')[:200]})")

    return "\n\n".join(lines)
