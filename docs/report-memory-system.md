# Hermes Agent Memory System - Technical Report

## Executive Summary

Hermes Agent implements a **multi-layered memory architecture** spanning seven distinct subsystems, each optimized for a different temporal scope and access pattern. The design balances always-available context (system prompt injection) with unbounded long-term recall (SQLite FTS5), procedural knowledge (skills), and optional AI-native user modeling (Honcho).

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              SYSTEM PROMPT                  │
                        │  ┌─────────────┐  ┌──────────────────┐     │
                        │  │ MEMORY.md   │  │   USER.md        │     │
                        │  │ (2200 chars)│  │   (1375 chars)   │     │
                        │  │ agent notes │  │   user profile   │     │
                        │  └──────┬──────┘  └────────┬─────────┘     │
                        │         │  Frozen Snapshot  │               │
                        │  ┌──────┴──────────────────┴─────────┐     │
                        │  │     Skills Index (available_skills) │     │
                        │  └───────────────────────────────────┘     │
                        │  ┌───────────────────────────────────┐     │
                        │  │  Honcho User Context (optional)   │     │
                        │  └───────────────────────────────────┘     │
                        │  ┌───────────────────────────────────┐     │
                        │  │  Context Files (SOUL.md, AGENTS.md)│     │
                        │  └───────────────────────────────────┘     │
                        └─────────────────────────────────────────────┘
                                           │
                                    LLM Inference
                                           │
                        ┌──────────────────┼──────────────────────────┐
                        │                  │                          │
                   ┌────▼────┐      ┌──────▼──────┐        ┌────────▼────────┐
                   │ memory  │      │session_search│        │  skill_manage   │
                   │  tool   │      │    tool      │        │     tool        │
                   └────┬────┘      └──────┬──────┘        └────────┬────────┘
                        │                  │                         │
              ┌─────────▼──────┐   ┌───────▼────────┐    ┌──────────▼────────┐
              │ ~/.hermes/     │   │ ~/.hermes/      │    │ ~/.hermes/skills/ │
              │  memories/     │   │  state.db       │    │   SKILL.md files  │
              │  MEMORY.md     │   │  (SQLite+FTS5)  │    │   per category    │
              │  USER.md       │   │                 │    │                   │
              └────────────────┘   └─────────────────┘    └───────────────────┘
```

---

## Layer 1: Persistent Curated Memory (MEMORY.md + USER.md)

**Source:** `tools/memory_tool.py` | **Storage:** `~/.hermes/memories/`

### Design

The primary memory system uses two bounded, file-backed stores injected into the system prompt at session start.

| Store        | Purpose                          | Char Limit | ~Token Budget |
|--------------|----------------------------------|------------|---------------|
| `MEMORY.md`  | Agent's personal notes           | 2,200      | ~800 tokens   |
| `USER.md`    | User profile & preferences       | 1,375      | ~500 tokens   |

### Frozen Snapshot Pattern

```
Session Start
    │
    ▼
┌──────────────────────────────────────────────────┐
│  load_from_disk()                                │
│  ├── Read MEMORY.md → memory_entries             │
│  ├── Read USER.md → user_entries                 │
│  ├── Deduplicate (preserve order, keep first)    │
│  └── Capture _system_prompt_snapshot (FROZEN)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  System prompt includes snapshot                  │
│  (never changes mid-session)                     │
│  → Preserves LLM prefix cache                   │
└──────────────────────────────────────────────────┘
    │
    ▼ (during session)
┌──────────────────────────────────────────────────┐
│  memory tool calls mutate live state             │
│  ├── Entries updated in-memory                   │
│  ├── Persisted to disk immediately (atomic)      │
│  └── Tool responses show live state              │
│  BUT system prompt snapshot unchanged            │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Next session start                              │
│  └── Snapshot refreshes from disk                │
└──────────────────────────────────────────────────┘
```

**Why frozen snapshots?** Changing the system prompt mid-session invalidates the LLM's KV prefix cache, causing all prior tokens to be reprocessed. The frozen snapshot avoids this cost.

### Entry Format

Entries use `§` (section sign) delimiters and can be multiline:

```
User's project is a Rust web service using Axum + SQLx
§
This machine runs Ubuntu 22.04, Docker and Podman installed
§
User prefers concise responses, dislikes verbose explanations
```

### Operations

| Action    | Behavior                                                        |
|-----------|-----------------------------------------------------------------|
| `add`     | Append entry. Rejects if total would exceed char limit.         |
| `replace` | Find entry by unique substring (`old_text`), replace entirely.  |
| `remove`  | Find entry by unique substring (`old_text`), delete it.         |

**Substring matching:** `old_text` only needs to uniquely identify one entry. If it matches multiple distinct entries, the operation fails with a disambiguation request.

### Atomic File Writes

```python
# Write to temp file in same directory (same filesystem)
fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
with os.fdopen(fd, "w") as f:
    f.write(content)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, str(path))  # Atomic on same filesystem
```

Readers always see either the old complete file or the new complete file -- never a partial write.

### Security Scanning

Every memory entry is scanned before acceptance for:

- **Prompt injection:** "ignore previous instructions", role hijacking, disregard rules
- **Exfiltration:** curl/wget with secrets, reading credential files
- **Persistence attacks:** SSH authorized_keys, shell RC modifications
- **Invisible Unicode:** Zero-width characters used for injection

Blocked content returns an error describing the matched threat pattern.

### Capacity Management

When memory exceeds 80% capacity, the agent consolidates entries:

```
Before (3 entries, ~90% full):
  "User runs macOS 14" (20 chars)
  "User has Homebrew"   (17 chars)
  "User has Docker"     (15 chars)

After consolidation (1 entry, ~40% full):
  "User runs macOS 14 Sonoma, Homebrew, Docker Desktop" (52 chars)
```

---

## Layer 2: Session Database (SQLite + FTS5)

**Source:** `hermes_state.py` | **Storage:** `~/.hermes/state.db`

### Schema

```
┌──────────────────────┐      ┌────────────────────────────┐
│      sessions        │      │         messages            │
├──────────────────────┤      ├────────────────────────────┤
│ id (PK)              │◄─────│ session_id (FK)            │
│ source               │      │ id (PK, autoincrement)     │
│ user_id              │      │ role                       │
│ model                │      │ content                    │
│ model_config (JSON)  │      │ tool_call_id               │
│ system_prompt        │      │ tool_calls (JSON)          │
│ parent_session_id(FK)│      │ tool_name                  │
│ started_at           │      │ timestamp                  │
│ ended_at             │      │ token_count                │
│ end_reason           │      │ finish_reason              │
│ message_count        │      └────────────────────────────┘
│ tool_call_count      │
│ input_tokens         │      ┌────────────────────────────┐
│ output_tokens        │      │     messages_fts (FTS5)    │
└──────────────────────┘      │     (virtual table)        │
                              │     content-backed from    │
                              │     messages table         │
                              └────────────────────────────┘
```

### Key Design Decisions

- **WAL mode** for concurrent readers + one writer (gateway multi-platform)
- **FTS5** virtual table with triggers for automatic index maintenance
- **Session chaining** via `parent_session_id` for compression splits
- **Source tagging** ('cli', 'telegram', 'discord', etc.) for filtering

### Session Lifecycle During Compression

```
Session A (active) ──compression──► Session A (ended, reason="compression")
                                          │
                                          ▼ parent_session_id
                                    Session B (new, continues conversation)
```

---

## Layer 3: Session Search (Long-Term Recall)

**Source:** `tools/session_search_tool.py`

### Flow

```
User asks about past conversation
           │
           ▼
┌──────────────────────────────┐
│  1. FTS5 search in state.db │
│     (ranked by relevance)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  2. Group by session, take   │
│     top N unique sessions    │
│     (default: 3)             │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  3. Load each session,       │
│     truncate ~100k chars     │
│     centered on matches      │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  4. Summarize via Gemini     │
│     Flash (auxiliary model)  │
│     ~10k token budget        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  5. Return per-session       │
│     summaries + metadata     │
└──────────────────────────────┘
```

### Memory vs Session Search

| Dimension        | Persistent Memory         | Session Search              |
|------------------|---------------------------|-----------------------------|
| **Capacity**     | ~1,300 tokens total       | Unlimited (all sessions)    |
| **Speed**        | Instant (in system prompt)| Search + LLM summarization  |
| **Use case**     | Key facts always present  | Recalling past conversations|
| **Management**   | Agent-curated             | Automatic                   |
| **Token cost**   | Fixed per session         | On-demand                   |

---

## Layer 4: Honcho Integration (AI-Native User Modeling)

**Source:** `honcho_integration/` | **External service:** [honcho.dev](https://honcho.dev)

Optional layer providing AI-generated user understanding that works across tools and sessions.

```
┌──────────────────────────────────────────┐
│           Each Conversation Turn         │
│                                          │
│  ┌─────────────┐     ┌───────────────┐  │
│  │  Prefetch:  │     │  Post-turn:   │  │
│  │  Get user   │     │  Sync msgs    │  │
│  │  context    │     │  to Honcho    │  │
│  │  → inject   │     │              │  │
│  │  into sys   │     │              │  │
│  │  prompt     │     │              │  │
│  └─────────────┘     └───────────────┘  │
│                                          │
│  ┌─────────────┐                        │
│  │ Tool:       │                        │
│  │ query_user_ │                        │
│  │ context     │                        │
│  └─────────────┘                        │
└──────────────────────────────────────────┘
```

Honcho runs alongside existing USER.md -- not a replacement. All calls are non-fatal.

---

## Layer 5: Skills (Procedural Memory)

**Source:** `tools/skill_manager_tool.py`, `tools/skills_tool.py`

Skills are the agent's procedural memory -- they capture *how to do a specific type of task* based on proven experience.

### Directory Structure

```
~/.hermes/skills/
├── my-skill/
│   ├── SKILL.md              # Instructions + frontmatter
│   ├── references/           # Reference docs
│   ├── templates/            # Code/config templates
│   ├── scripts/              # Automation scripts
│   └── assets/               # Images, data files
└── category-name/
    └── another-skill/
        └── SKILL.md
```

### Skill Lifecycle

```
Complex task completed (5+ tool calls)
           │
           ▼
┌──────────────────────────────┐
│  Skill nudge fires           │
│  (every 15 iterations)       │
│  "[System: consider saving   │
│   this as a skill]"          │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Agent creates skill via     │
│  skill_manage(action=create) │
│  ├── Validate name, content  │
│  ├── Parse YAML frontmatter  │
│  ├── Security scan           │
│  └── Write to disk           │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Next session: skill index   │
│  built into system prompt    │
│  ├── Scans ~/.hermes/skills/ │
│  ├── Groups by category      │
│  ├── Platform compatibility  │
│  └── Agent matches by meaning│
└──────────────────────────────┘
```

### Security

Agent-created skills undergo the same security scanning as community hub installs, preventing injection via skill content.

---

## Layer 6: Todo Store (In-Memory Task Planning)

**Source:** `tools/todo_tool.py`

An in-memory, per-session task list for decomposing complex work.

```
┌──────────────────────────────────────────┐
│  TodoStore (in-memory, one per session)  │
│  ├── items: [{id, content, status}]      │
│  ├── write(todos, merge=False|True)      │
│  ├── read() → current list               │
│  └── format_for_injection()              │
│       └── Survives context compression   │
└──────────────────────────────────────────┘
```

The todo state is:
- **Hydrated from history** when the gateway creates a fresh AIAgent per message
- **Injected after compression** so task context survives summarization
- **Not persisted to disk** -- session-scoped only

---

## Layer 7: Context Compression (Memory Management)

**Source:** `agent/context_compressor.py`

When conversation approaches the model's context limit, the compressor summarizes middle turns while preserving critical context.

```
┌─────────────────────────────────────────────────────┐
│              Conversation Messages                   │
│                                                      │
│  ┌──────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ Protected │  │   SUMMARIZED    │  │ Protected  │ │
│  │ First 3   │  │   Middle turns  │  │ Last 4     │ │
│  │ turns     │  │   → single msg  │  │ turns      │ │
│  └──────────┘  └─────────────────┘  └────────────┘ │
│                                                      │
│  Pre-compression:                                    │
│  ├── Memory flush (save important info)             │
│  ├── Todo state captured                            │
│  Post-compression:                                   │
│  ├── Orphaned tool pairs sanitized                  │
│  ├── Todo state re-injected                         │
│  ├── System prompt rebuilt (memory reloaded)        │
│  └── Session split in SQLite (parent→child chain)   │
└─────────────────────────────────────────────────────┘
```

---

## Proactive Memory Behaviors

### Memory Nudge System

```python
# Every N user turns (default 10), inject a reminder:
"[System: You've had several exchanges in this session.
 Consider whether there's anything worth saving to your memories.]"
```

### Memory Flush Before Compression

```python
# Before context is compressed (and middle turns lost):
"[System: The session is being compressed.
 Please save anything worth remembering to your memories.]"
# → One LLM call with only the memory tool available
# → All flush artifacts stripped from messages afterward
```

### Skill Creation Nudge

```python
# After 15+ tool-calling iterations:
"[System: The previous task involved many steps.
 If you discovered a reusable workflow, consider saving it as a skill.]"
```

---

## Data Flow Summary

```
                    SESSION START
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
  Load MEMORY.md   Load USER.md      Build Skills Index
  Load from disk   Load from disk    Scan ~/.hermes/skills/
        │               │                   │
        └───────┬───────┘                   │
                ▼                           │
         Freeze snapshot                    │
                │                           │
                └───────────┬───────────────┘
                            ▼
                    SYSTEM PROMPT
                    (frozen for session)
                            │
                            ▼
                   ┌─ CONVERSATION ─┐
                   │                │
                   │  memory tool ──┼──► Disk write (atomic)
                   │  session_search┼──► SQLite FTS5 → Gemini summary
                   │  skill_manage ─┼──► ~/.hermes/skills/ write
                   │  todo ─────────┼──► In-memory store
                   │  honcho ───────┼──► External API (optional)
                   │                │
                   │  Context full? ┼──► Memory flush → Compress
                   │                │    → Split session → Reload
                   └────────────────┘
                            │
                            ▼
                    SESSION END
                    ├── Messages → SQLite (state.db)
                    ├── Messages → JSON log
                    └── Memory on disk ready for next session
```

---

## Configuration

```yaml
# ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200      # ~800 tokens
  user_char_limit: 1375        # ~500 tokens
  nudge_interval: 10           # User turns between memory nudges
  flush_min_turns: 6           # Min turns before compression flush

skills:
  creation_nudge_interval: 15  # Tool iterations before skill nudge

# Environment variables
CONTEXT_COMPRESSION_THRESHOLD=0.85   # % of context limit
CONTEXT_COMPRESSION_ENABLED=true
CONTEXT_COMPRESSION_MODEL=           # Override summarization model
```

---

## Key Files

| File | Purpose |
|------|---------|
| `tools/memory_tool.py` | MemoryStore class, security scanning, tool schema |
| `hermes_state.py` | SessionDB with SQLite + FTS5 |
| `tools/session_search_tool.py` | Long-term recall via FTS5 + LLM summarization |
| `tools/skill_manager_tool.py` | Skill CRUD with security scanning |
| `tools/todo_tool.py` | In-memory task planning |
| `agent/context_compressor.py` | Context window compression |
| `agent/prompt_builder.py` | System prompt assembly (memory, skills, context files) |
| `honcho_integration/` | Optional AI-native user modeling |
| `run_agent.py` | AIAgent orchestration (memory init, nudges, flush, compression) |
