# ctxpact

**Resilient context compaction proxy for local LLM inference.**

ctxpact is a lightweight, OpenAI-compatible proxy that handles oversized inputs. When a 110k-token document hits a 16k-token model, ctxpact extracts the most relevant ~12k tokens to answer accurately — achieving **100% on our reading comprehension benchmark** with the right model+strategy.

Drop it in front of any llama.cpp / Ollama / vLLM / vLLM-mlx server. Works with any agentic framework ( OpenClaw, Hermes, etc.) that speaks the OpenAI API.

## How it works

ctxpact sits between your agent and your local LLM:

```
Agent (OpenClaw, Hermes, etc.)
  │
  ▼
ctxpact proxy (localhost:8000)       ◄── OpenAI-compatible, drop-in
  │
  ├── Stage 1: DCP — dedup tool calls, strip stale writes, truncate errors
  ├── Stage 2: Summarize — evict old context, keep recent turns
  ├── Stage 3: Extract — 16 strategies to pull relevant content
  │
  ▼
Local LLM (llama-server / Ollama / vLLM)
```

**No API keys. No cloud. Everything runs on your hardware.**

### Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, streaming
- **3-stage compaction pipeline** + **16 extraction strategies** — from zero-LLM-call heuristics (`header`, `embed`, `autosearch`) to dual-LLM retrieval pipelines (`readagent`, `agentic`). Pick via `compaction.oversized.strategy`
- **Provider failover** — circuit breaker + health monitoring across multiple backends
- **Session tracking** — per-conversation state with a compaction audit trail
- **Handoff on compaction (opt-in)** — when ctxpact compacts your conversation, the model is instructed to warn you and give you a paste-ready summary you can start a new chat with (once per chat; a short reminder on the second compaction). Enable with `compaction.handoff.enabled: true`

## Quick Start

```bash
# Install
git clone https://github.com/user/ctxpact && cd ctxpact
pip install -e .

# Start your LLM backend (example: llama-server)
llama-server -m Qwen3.5-9B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --ctx-size 16384 --jinja -ngl 99

# Start ctxpact
python -m ctxpact.server --config config.yaml --local --strategy readagent

# Use it — same API as your LLM, just different port
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-Q8_0.gguf", "messages": [{"role": "user", "content": "...your 100k token message..."}]}'
```

## Benchmark Results

Tested on Frankenstein (110k tokens → 12k budget, 8 reading comprehension questions) and LoCoMo-MC10 (multi-session conversation QA, 10-choice, 20 questions):

| Configuration | Frankenstein | LoCoMo-MC10 | Combined |
|---------------|-------------|-------------|----------|
| **readagent + Qwen3.5-9B** | **8/8 (100%)** | **15/20 (75%)** | **87.5%** |
| rlm + Qwen3.5-9B | 8/8 (100%) | 12/20 (60%) | 80.0% |
| embed + Qwen3.5-9B | 7/8 (87.5%) | 14/20 (70%) | 78.8% |
| agentic + LFM2-8B-A1B | 6.2/8 avg (78%) | 5/20 (25%) | 51.3% |
| Random baseline (LoCoMo) | — | 2/20 (10%) | — |

`readagent` is the recommended default. **Model choice matters more than strategy** — NR-MMLU (reading comprehension) is the best predictor of performance. Full analysis: [BENCHMARKS.md](BENCHMARKS.md).

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **Qwen3.5-9B (Q8_0)** | 11 tok/s | 8/8 Frankenstein, 75% LoCoMo | **Best accuracy** |
| LFM2-8B-A1B (Q8_0) | 50 tok/s | 6.2/8 Frankenstein, 25% LoCoMo | Fast inference |

## Configuration

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

providers:
  - name: "local"
    url: "http://localhost:8080/v1"     # Your llama-server
    model: "Qwen3.5-9B-Q8_0.gguf"
    api_key: "dummy"
    max_context: 16384
    priority: 1
    timeout_seconds: 180

circuit_breaker:
  failure_threshold: 3
  recovery_timeout_seconds: 30

compaction:
  enabled: true
  triggers:
    token_ratio: 0.70                   # Compact when input > 70% of context

  stage1_dcp:                           # Dynamic Context Pruning
    dedup_tool_calls: true
    strip_superseded_writes: true
    truncate_errors: true

  stage2_summarize:                     # LLM-based summarization
    max_summary_tokens: 2000
    retention_window: 6

  oversized:
    strategy: "readagent"               # Which extraction strategy to use

  handoff:
    enabled: false                      # true = on compaction, the model warns you
                                        # and gives a summary to start a new chat
```

## API

```
GET  /health                    # Provider status + compaction config
GET  /v1/models                 # OpenAI-compatible model listing
POST /v1/chat/completions       # Main endpoint (streaming supported)
```

Session tracking via the `X-Session-ID` header (auto-generated if not provided). Inspect a session at `GET /v1/sessions/{session_id}`.

## Development

```bash
make test          # Run pytest
make lint          # Run ruff
make run           # Start server with default config
make docker-build  # Build container
```

## License

MIT
