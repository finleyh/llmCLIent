# cli-mcp-client

A Metasploit-styled REPL (`mcpc`) for chatting with a remote, OpenAI-compatible LLM
while configuring and using stdio + SSE MCP servers. Built as a `uv` project.

## How memory works

Three layers, mirroring desktop chat clients:

1. **Short-term (conversation):** the LLM is stateless, so the full message list
   is re-sent on every API call. When it exceeds the context budget, the oldest
   messages are dropped (`llm.trim_history`). Trimming subtracts the `tools` schema
   size (significant with large MCP servers) from the budget, drops messages by
   *block* so an assistant `tool_calls` message and its results are never split, and
   truncates oversized tool outputs in the re-sent copy (the full output stays in
   SQLite). System messages are always kept.
2. **Long-term (cross-session):** durable facts are stored in SQLite and injected
   into the system prompt at the start of every turn. The model writes them by
   calling the built-in `save_memory` tool; you can also use `memory add`.
3. **Sessions:** every conversation is persisted to SQLite so you can `session
   resume <id>` later — like Metasploit workspaces.

## Setup

```bash
cp .env.example .env      # fill in LLM_BASE_URL, LLM_MODEL, LLM_AUTH_TOKEN
uv sync
uv run mcpc
```

## Config (.env)

| var | meaning |
|-----|---------|
| `LLM_BASE_URL` | OpenAI-compatible base, e.g. `https://api.openai.com/v1` |
| `LLM_MODEL` | model id |
| `LLM_AUTH_TOKEN` | bearer token / API key |
| `LLM_MAX_CONTEXT_TOKENS` | context window; history is trimmed to fit (default 128000) |
| `LLM_MAX_TOOL_OUTPUT_TOKENS` | cap per tool-result when re-sending history; `0` disables (default 4000) |
| `LLM_TEMPERATURE` | sampling temperature (default 0.7) |
| `MCPC_DB_PATH` | SQLite location (default `~/.mcpc/mcpc.db`) |

## Commands

```
chat <msg>            send a message (or just type text)
sessions              list saved sessions
session new [name]    start a new session
session resume <id>   load a past session
memory list|add|del   manage long-term memory
mcp add stdio <name> <cmd> [args...]
mcp add sse   <name> <url>
mcp connect <name>    connect a server
mcp tools             list connected tools
set                   show LLM config
exit                  quit
```

## Example: add a local stdio MCP server

```
mcpc> mcp add stdio files npx -y @modelcontextprotocol/server-filesystem /tmp
mcpc> mcp connect files
mcpc> mcp tools
mcpc> chat list the files in /tmp
```

## Example: connect HexStrike AI (150+ security tools)

[HexStrike AI](https://github.com/0x4m4/hexstrike-ai) is itself an MCP server, so this
client acts as its host — the same role Claude Desktop plays. HexStrike runs as **two
processes**: an execution engine (`hexstrike_server.py`, an HTTP server on `:8888` that
wraps the actual tools) and a thin MCP adapter (`hexstrike_mcp.py`) that speaks MCP over
stdio and forwards to the engine.

```
this REPL  ──MCP/stdio──>  hexstrike_mcp.py  ──HTTP──>  hexstrike_server.py  ──>  nmap/nuclei/…
(LLM host)                 (MCP adapter)                (:8888 engine)            (real tools)
```

1. Start the engine in a separate terminal (per HexStrike's install docs) and confirm it's
   reachable at `http://localhost:8888`:

   ```bash
   python3 hexstrike_server.py
   ```

2. Register the adapter as a stdio server in this client and connect:

   ```
   mcpc> mcp add stdio hexstrike python3 /abs/path/hexstrike_mcp.py --server http://localhost:8888
   mcpc> mcp connect hexstrike
   mcpc> mcp tools
   mcpc> chat scan localhost for open ports
   ```

The adapter's tools are exposed to the remote LLM as `hexstrike__<tool>` functions and the
tool-call loop dispatches them automatically — no code changes required.

**Caveats**

- **Tool count (known trade-off):** HexStrike exposes 150+ tools, and every connected
  tool's schema is sent to the LLM on each call. This is a lot of tokens and can degrade
  tool selection and response quality on smaller models. The client intentionally exposes
  all tools — HexStrike is designed to operate with its full toolset available — so this is
  an accepted, known limitation rather than something to mitigate by disabling tools. Use a
  model with a large context window and strong tool-calling for best results.
- **Transport:** the stdio adapter above is the reliable path. HexStrike's engine also
  exposes an MCP endpoint at `:8888/mcp`, but that is the newer *streamable-HTTP* transport,
  whereas `mcp add sse` here uses the older *SSE* transport and may not connect to it. Use
  stdio.
- **Authorization:** these are live offensive security tools. Only run them against systems
  you are explicitly authorized to test.
