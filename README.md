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
| `AGENT_MAX_STEPS` | hard cap on agent reason→act→observe iterations (default 25) |
| `AGENT_AUTO_APPROVE_ALL` | `1` = fully autonomous; never confirm before a tool runs (default `0`) |
| `AGENT_READONLY_PREFIXES` | tool-name prefixes auto-approved in hybrid mode (overrides the default verb list) |
| `AGENT_ALLOW` | comma-separated exact tool names to always auto-approve |
| `AGENT_DENY` | comma-separated exact tool names to always block (overrides allow) |

## Commands

```
chat <msg>            send a message (or just type text)
sessions              list saved sessions
session new [name]    start a new session
session resume <id>   load a past session
memory list|add|del   manage long-term memory
agent run <objective> run an autonomous ReAct agent toward a goal
agent runs            list past agent runs
agent show <id>       show the step-by-step trace of a run
mcp add stdio <name> <cmd> [args...]
mcp add sse   <name> <url>
mcp connect <name>    connect a server
mcp tools             list connected tools
set                   show LLM config
exit                  quit
```

## Agent mode

`chat` is a single conversational turn (the model may chain tool calls, but you drive each
turn). `agent run <objective>` instead hands the model a **goal** and lets it loop —
reason, call a tool, read the result, repeat — until it calls the built-in `task_complete`
tool or hits `AGENT_MAX_STEPS`.

**Approval is hybrid by default.** Read-only tools (names starting `get`, `list`, `search`,
`read`, …) run automatically; anything that looks state-changing pauses for confirmation:

```
mcpc> agent run find open ports on localhost and save a summary to notes.md
⚠ approval needed srv__write_file (state-changing tool)
  args: {"path": "notes.md", ...}
  approve? [y]es / [a]lways / [s]kip / a[b]ort >
```

`always` auto-approves that tool for the rest of the run; `skip` feeds a "did not run" note
back to the model; `abort` (or Ctrl-C) ends the run. Set `AGENT_AUTO_APPROVE_ALL=1` for
fully autonomous runs, or use `AGENT_ALLOW` / `AGENT_DENY` for per-tool overrides.

Every run is persisted: `agent runs` lists them and `agent show <id>` replays the full
thought → tool → observation → final trace. Runs share the active session's history, so the
agent sees prior conversation and long-term memories, and what it does is visible to later
`chat` turns.

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

**You must start HexStrike's engine before this client can connect to it.** `mcp connect`
only launches the adapter (`hexstrike_mcp.py`); the adapter immediately tries to reach the
engine over HTTP, so the engine has to be running and healthy first.

### Step 1 — install and start the HexStrike engine

Per HexStrike's own docs, in a **separate terminal**:

```bash
git clone https://github.com/0x4m4/hexstrike-ai.git
cd hexstrike-ai
python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
pip3 install -r requirements.txt

# start the engine (defaults to port 8888; --debug for verbose logs)
python3 hexstrike_server.py
```

Leave this running. (The actual security tools — nmap, nuclei, gobuster, etc. — must also
be installed on that machine; the engine wraps whatever is present.)

### Step 2 — confirm the engine is up

```bash
curl http://localhost:8888/health
```

Don't move on until this returns a healthy response. If it fails, the adapter in step 3
will fail too.

### Step 3 — connect from this client

Note the **absolute path** to `hexstrike_mcp.py` inside the repo you cloned (and use the
same venv's Python so its dependencies resolve):

```
mcpc> mcp add stdio hexstrike /abs/path/hexstrike-ai/hexstrike-env/bin/python3 /abs/path/hexstrike-ai/hexstrike_mcp.py --server http://localhost:8888
mcpc> mcp connect hexstrike
mcpc> mcp tools
mcpc> chat scan localhost for open ports
```

`mcp connect` spawns the adapter, which initializes against the running engine and reports
its tools. They're exposed to the remote LLM as `hexstrike__<tool>` functions and the
tool-call loop dispatches them automatically — no code changes required.

> If the engine runs on a different host (e.g. on your VM host while the client runs in a
> guest), point `--server` at that address instead of `localhost`, e.g.
> `--server http://192.168.56.1:8888`.

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
