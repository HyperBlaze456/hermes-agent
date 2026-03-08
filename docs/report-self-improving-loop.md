# Hermes Agent Self-Improving Loop - Technical Report

## Executive Summary

Hermes Agent implements a **multi-mechanism self-improving loop** that operates at three temporal scales: **intra-session** (error recovery, context management, task planning), **inter-session** (persistent memory, skills, session search), and **training-time** (RL via Tinker-Atropos, trajectory compression). The system combines reactive self-correction (retries, recovery messages) with proactive self-improvement (memory nudges, skill creation, context consolidation) to create an agent that gets better at both individual tasks and repeated task patterns.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVING LOOP                                  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  TRAINING-TIME IMPROVEMENT (offline)                             │  │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │  │
│  │  │ Trajectory  │  │  Trajectory      │  │  RL Training     │   │  │
│  │  │ Capture     │──│  Compression     │──│  (Tinker-Atropos)│   │  │
│  │  │ (JSONL)     │  │  (budget fit)    │  │  GRPO / PPO      │   │  │
│  │  └─────────────┘  └──────────────────┘  └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  INTER-SESSION IMPROVEMENT (persistent)                          │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ Memory   │  │  Skills   │  │ Session  │  │  Honcho      │  │  │
│  │  │ MEMORY.md│  │ Procedural│  │  Search  │  │  User Model  │  │  │
│  │  │ USER.md  │  │ Knowledge │  │  (FTS5)  │  │  (optional)  │  │  │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  INTRA-SESSION IMPROVEMENT (reactive)                            │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ Error    │  │  Context  │  │  Task    │  │  Subagent    │  │  │
│  │  │ Recovery │  │  Compress │  │  Planning│  │  Delegation  │  │  │
│  │  │ & Retry  │  │  & Adapt  │  │  (Todo)  │  │  (Parallel)  │  │  │
│  │  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Agent Loop

**Source:** `run_agent.py:run_conversation()` (line ~2860)

The main conversation loop is the backbone of all improvement mechanisms.

### Loop Structure

```
run_conversation(user_message)
        │
        ├── Reset retry counters & iteration budget
        ├── Hydrate todo store from history
        ├── Memory nudge injection (if interval reached)
        ├── Skill creation nudge injection (if interval reached)
        ├── Honcho prefetch (user context)
        ├── Build/cache system prompt
        │
        ├── PREFLIGHT COMPRESSION
        │   └── If loaded history exceeds context threshold
        │       └── Multi-pass compression (up to 3 passes)
        │
        ▼
┌─── MAIN LOOP (while iterations < max && budget > 0) ──────┐
│                                                             │
│  ┌─ API CALL ─────────────────────────────────────────┐    │
│  │  Build api_messages (strip reasoning, add system)  │    │
│  │  Apply prompt caching (Claude via OpenRouter)      │    │
│  │  Sanitize orphaned tool pairs                      │    │
│  │  Call LLM via OpenAI-compatible API                │    │
│  └────────────────────┬───────────────────────────────┘    │
│                       │                                     │
│              ┌────────▼──────────┐                          │
│              │ Response valid?   │                          │
│              └──┬────────────┬───┘                          │
│            Yes  │            │ No                           │
│                 │     ┌──────▼────────────┐                 │
│                 │     │ ERROR RECOVERY    │                 │
│                 │     │ (see Section 2)   │                 │
│                 │     └──────────────────┘                 │
│                 ▼                                           │
│        ┌────────────────┐                                   │
│        │ Has tool_calls?│                                   │
│        └──┬──────────┬──┘                                   │
│       Yes │          │ No (text response)                   │
│           │          │                                      │
│           ▼          ▼                                      │
│   ┌──────────┐  ┌──────────────┐                           │
│   │ Validate │  │ Return final │                           │
│   │ tools    │  │ response     │                           │
│   │ & args   │  └──────────────┘                           │
│   └────┬─────┘                                              │
│        │                                                    │
│        ▼                                                    │
│   ┌──────────────┐                                          │
│   │ Execute tool  │                                         │
│   │ calls         │                                         │
│   └────┬─────────┘                                          │
│        │                                                    │
│        ▼                                                    │
│   ┌──────────────┐     ┌──────────────────┐                │
│   │ Context full?│────►│ Compress & split │                │
│   └──────────────┘ Yes │ session          │                │
│        │ No            └──────────────────┘                │
│        │                                                    │
│        └──── LOOP BACK ─────────────────────────────────►  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  MAX ITERATIONS REACHED?
        │
        ▼
  Request summary from LLM → Return
```

### Iteration Budget

```python
class IterationBudget:
    """Thread-safe shared counter for parent + all child agents."""

    def __init__(self, max_total: int):     # Default: 90
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:   # Returns False if exhausted
    def refund(self) -> None:    # execute_code calls refunded
```

The budget is shared across the parent and all delegated subagents, preventing runaway chains. `execute_code` iterations (RPC-style programmatic tool calls) are refunded as they're cheap.

---

## 2. Error Recovery & Self-Correction

The agent implements **six distinct error recovery strategies**, each targeting a specific failure mode.

### Recovery Strategy Map

```
┌───────────────────────────────────────────────────────────────────────┐
│                     ERROR RECOVERY HIERARCHY                          │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  1. INVALID TOOL CALLS (hallucinated tool names)               │ │
│  │     ├── Detect: tc.function.name not in valid_tool_names       │ │
│  │     ├── Action: Silent retry (don't add anything to messages)  │ │
│  │     ├── Limit: 3 retries                                       │ │
│  │     └── Fallback: Return partial result                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  2. INVALID JSON ARGUMENTS                                     │ │
│  │     ├── Detect: json.loads(args) fails                         │ │
│  │     ├── Action: Silent retry (3x), then inject recovery msg    │ │
│  │     ├── Recovery msg: "Your tool call had invalid JSON...      │ │
│  │     │   Please retry with valid JSON or respond without it."   │ │
│  │     └── Self-healing: Empty strings auto-fixed to "{}"         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  3. INVALID/EMPTY API RESPONSES                                │ │
│  │     ├── Detect: None response, no choices, empty choices       │ │
│  │     ├── Action: Exponential backoff (5s, 10s, 20s... 120s)     │ │
│  │     ├── Limit: 6 retries                                       │ │
│  │     ├── Interrupt-aware: Checks between sleep intervals        │ │
│  │     └── Diagnostics: Provider name, error msg, response time   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  4. CONTEXT LENGTH EXCEEDED                                    │ │
│  │     ├── Detect: Error msg contains "context length", etc.      │ │
│  │     ├── Action A: Parse actual limit from error → step down    │ │
│  │     ├── Action B: get_next_probe_tier() → reduce context       │ │
│  │     ├── Action C: Compress conversation + retry                │ │
│  │     ├── Limit: 3 compression attempts                          │ │
│  │     └── Caches discovered context length for future sessions   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  5. PAYLOAD TOO LARGE (413)                                    │ │
│  │     ├── Detect: HTTP 413, "request entity too large"           │ │
│  │     ├── Action: Compress conversation + retry                  │ │
│  │     ├── Limit: 3 compression attempts                          │ │
│  │     └── Separate from context-length (different root cause)    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  6. TRUNCATED RESPONSES (finish_reason='length')               │ │
│  │     ├── Detect: finish_reason == "length"                      │ │
│  │     ├── Action: Roll back to last complete assistant turn      │ │
│  │     ├── For Codex: Continuation retries (up to 3x)             │ │
│  │     └── First message truncated: Return failed (no rollback)   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  7. AUTH FAILURES (401)                                        │ │
│  │     ├── Detect: HTTP 401 from Codex or Nous provider           │ │
│  │     ├── Action: Refresh credentials + retry (once per type)    │ │
│  │     └── Codex: _try_refresh_codex_client_credentials(force)    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

### Context Adaptation Flow

```
API Error: "context length exceeded"
            │
            ▼
┌──────────────────────────────┐
│ Parse actual limit from      │
│ error message                │
│ (e.g., "max 32768 tokens")  │
└──────────┬───────────────────┘
           │
     Found?│
    ┌──────┴──────┐
    │ Yes         │ No
    ▼             ▼
┌────────┐  ┌────────────────┐
│ Use    │  │ Step down to   │
│ parsed │  │ next probe tier│
│ limit  │  │ (predefined    │
│        │  │  ladder)       │
└───┬────┘  └──────┬─────────┘
    │              │
    └──────┬───────┘
           ▼
┌──────────────────────────────┐
│ Update compressor:           │
│ ├── context_length = new_ctx │
│ ├── threshold_tokens recalc  │
│ └── _context_probed = True   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Compress conversation        │
│ ├── Memory flush first       │
│ ├── Summarize middle turns   │
│ ├── Sanitize tool pairs      │
│ └── Inject todo snapshot     │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Cache discovered limit       │
│ (save_context_length)        │
│ for future sessions          │
└──────────────────────────────┘
```

---

## 3. Context Window Management

**Source:** `agent/context_compressor.py`

### Compression Algorithm

```
Input: [msg_0, msg_1, ..., msg_N]

1. PROTECT HEAD:  first 3 messages (system, user, first response)
2. PROTECT TAIL:  last 4 messages (recent actions/context)
3. ALIGN BOUNDARIES: don't split tool_call/result pairs
4. SUMMARIZE MIDDLE: via auxiliary model (Gemini Flash)
5. SANITIZE: fix orphaned tool_call/result pairs
6. INJECT: todo state snapshot after compressed messages

Output: [msg_0..msg_2, SUMMARY, msg_(N-3)..msg_N, TODO_SNAPSHOT]
```

### Trigger Points

```
┌────────────────────────────────────────┐
│         COMPRESSION TRIGGERS           │
│                                        │
│  1. PREFLIGHT (before loop starts)     │
│     └── Loaded history > threshold     │
│         (e.g., model switch to smaller │
│          context window)               │
│                                        │
│  2. POST-TOOL (in loop)               │
│     └── should_compress() returns True │
│         (prompt_tokens >= threshold)   │
│                                        │
│  3. ERROR-TRIGGERED                    │
│     └── Context-length error from API  │
│     └── 413 payload too large          │
│                                        │
│  Threshold: 85% of context_length     │
│  (configurable via env var)            │
└────────────────────────────────────────┘
```

### Pre-Compression Memory Flush

Before any compression event, the agent gets one chance to save important information:

```
1. Inject: "[System: The session is being compressed.
            Please save anything worth remembering.]"
2. API call with ONLY the memory tool available
3. Execute any memory tool calls returned
4. Strip ALL flush artifacts from messages
5. Proceed with compression
```

This ensures the agent doesn't lose critical information when middle turns are summarized.

---

## 4. Proactive Learning Mechanisms

### Memory Nudge Cycle

```
Turn 1  ──►  Turn 2  ──►  ...  ──►  Turn 10  ──►  Turn 11
                                        │              │
                                        ▼              │
                                   NUDGE FIRES         │
                                   "[System: Consider  │
                                    saving memories]"  │
                                        │              │
                                   Counter resets ◄────┘
                                   if memory tool used
```

### Skill Creation Cycle

```
Tool Iteration 1 ──► 2 ──► ... ──► 15 ──► Next user message
                                    │           │
                                    ▼           ▼
                            Counter = 15    NUDGE FIRES
                                            "[System: The
                                             previous task
                                             involved many
                                             steps. Consider
                                             saving as skill.]"
                                                │
                                           Counter resets
                                           if skill_manage used
```

### Knowledge Crystallization Pipeline

```
Experience (intra-session)
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
┌──────────────┐              ┌──────────────┐
│  Declarative │              │  Procedural  │
│  Knowledge   │              │  Knowledge   │
│              │              │              │
│  memory tool │              │ skill_manage │
│  → MEMORY.md │              │ → SKILL.md   │
│  → USER.md   │              │ + references │
│              │              │ + templates  │
│  "What I     │              │              │
│   know"      │              │ "How to do   │
│              │              │  a task"     │
└──────┬───────┘              └──────┬───────┘
       │                             │
       ▼                             ▼
  Next session:                Next session:
  In system prompt             In skills index
  (instant access)             (loaded on match)
```

---

## 5. Subagent Delegation

**Source:** `tools/delegate_tool.py`

Delegation enables parallel problem-solving with isolated context.

```
┌─────────────────────────────────────────────────────────┐
│  PARENT AGENT                                           │
│  ├── iteration_budget (shared, thread-safe)             │
│  │                                                      │
│  │  delegate_task(tasks=[...])                          │
│  │       │                                              │
│  │       ▼                                              │
│  │  ┌──────────────────────────────────────────────┐   │
│  │  │  ThreadPoolExecutor (max 3 concurrent)       │   │
│  │  │                                              │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │  │ Child 1  │ │ Child 2  │ │ Child 3  │    │   │
│  │  │  │          │ │          │ │          │    │   │
│  │  │  │ Fresh    │ │ Fresh    │ │ Fresh    │    │   │
│  │  │  │ context  │ │ context  │ │ context  │    │   │
│  │  │  │          │ │          │ │          │    │   │
│  │  │  │ Blocked: │ │ Blocked: │ │ Blocked: │    │   │
│  │  │  │ delegate │ │ delegate │ │ delegate │    │   │
│  │  │  │ clarify  │ │ clarify  │ │ clarify  │    │   │
│  │  │  │ memory   │ │ memory   │ │ memory   │    │   │
│  │  │  │ send_msg │ │ send_msg │ │ send_msg │    │   │
│  │  │  │ exec_code│ │ exec_code│ │ exec_code│    │   │
│  │  │  └────┬─────┘ └────┬─────┘ └────┬─────┘    │   │
│  │  │       │            │            │           │   │
│  │  │       └────────┬───┘────────────┘           │   │
│  │  │                ▼                             │   │
│  │  │         Summary results                     │   │
│  │  └──────────────────────────────────────────────┘   │
│  │       │                                              │
│  │       ▼                                              │
│  │  Parent sees ONLY delegation call + summary          │
│  │  (never child's intermediate reasoning)              │
│  └──────────────────────────────────────────────────────┘
│                                                          │
│  Max depth: 2 (parent → child; grandchild rejected)     │
│  Default child iterations: 50                            │
│  Default child toolsets: terminal, file, web             │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Training-Time Improvement (RL Loop)

**Source:** `rl_cli.py`, `tools/rl_training_tool.py`, `trajectory_compressor.py`, `batch_runner.py`

### RL Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL TRAINING PIPELINE                          │
│                                                                  │
│  ┌──────────────┐     ┌──────────────────┐                      │
│  │ batch_runner  │     │ trajectory_      │                      │
│  │ .py           │     │ compressor.py    │                      │
│  │               │     │                  │                      │
│  │ Parallel      │────►│ Post-process:    │                      │
│  │ trajectory    │     │ ├── Protect head │                      │
│  │ generation    │     │ ├── Protect tail │                      │
│  │ (JSONL)       │     │ ├── Summarize    │                      │
│  │               │     │ │   middle turns │                      │
│  │               │     │ └── Fit token    │                      │
│  │               │     │     budget       │                      │
│  └──────────────┘     └────────┬─────────┘                      │
│                                │                                 │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TINKER-ATROPOS (RL Framework)               │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐ │   │
│  │  │ Environments │  │ Rollout     │  │ Training       │ │   │
│  │  │ (BaseEnv     │  │ Server      │  │ (GRPO/PPO)     │ │   │
│  │  │  subclasses) │  │ (sglang)    │  │                │ │   │
│  │  │              │  │             │  │ ├── LoRA r=32  │ │   │
│  │  │ AST-scanned  │  │ Qwen3-8B   │  │ ├── lr=4e-5   │ │   │
│  │  │ for discovery│  │ default     │  │ ├── 2500 steps │ │   │
│  │  └──────────────┘  └─────────────┘  │ └── WandB     │ │   │
│  │                                      │     metrics   │ │   │
│  │                                      └────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Agent interface (rl_* tools):                                  │
│  ├── rl_list_environments   (discover envs)                     │
│  ├── rl_select_environment  (choose env)                        │
│  ├── rl_get_current_config  (read config)                       │
│  ├── rl_edit_config         (modify config — guarded fields)    │
│  ├── rl_start_training      (launch subprocess)                 │
│  ├── rl_check_status        (poll WandB metrics)                │
│  ├── rl_stop_training       (terminate run)                     │
│  ├── rl_get_results         (fetch final metrics)               │
│  ├── rl_list_runs           (browse history)                    │
│  └── rl_test_inference      (validate post-training)            │
│                                                                  │
│  Locked fields (not editable by agent):                         │
│  ├── tokenizer_name, rollout_server_url                         │
│  ├── model_name, base_url, server_type                          │
│  ├── lora_rank, learning_rate, checkpoint_dir                   │
│  └── max_token_length, max_num_workers                          │
└─────────────────────────────────────────────────────────────────┘
```

### Trajectory Compression Strategy

```
Input: Full agent trajectory (may exceed training token budget)

┌────────────────────────────────────────────────────────┐
│  1. PROTECT FIRST TURNS                                │
│     system, human, first gpt, first tool               │
│     (task understanding + initial approach)             │
│                                                        │
│  2. PROTECT LAST N TURNS                               │
│     final actions and conclusions                      │
│     (correct answer + final reasoning)                 │
│                                                        │
│  3. COMPRESS MIDDLE (only as needed)                   │
│     starting from 2nd tool response                    │
│     → replace with single human summary message        │
│                                                        │
│  4. KEEP REMAINING TOOL CALLS INTACT                   │
│     model continues working after summary              │
│                                                        │
│  Target: fit under target_max_tokens (default 16k)     │
└────────────────────────────────────────────────────────┘
```

---

## 7. Complete Self-Improvement Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    COMPLETE IMPROVEMENT LIFECYCLE                        │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  SESSION N                                                    │     │
│   │                                                               │     │
│   │  1. LOAD PRIOR KNOWLEDGE                                     │     │
│   │     ├── MEMORY.md → system prompt                            │     │
│   │     ├── USER.md → system prompt                              │     │
│   │     ├── Skills index → system prompt                         │     │
│   │     ├── Session history → SQLite (searchable)                │     │
│   │     └── Honcho user model → system prompt (optional)         │     │
│   │                                                               │     │
│   │  2. EXECUTE WITH SELF-CORRECTION                              │     │
│   │     ├── Tool call validation (hallucination detection)       │     │
│   │     ├── JSON argument repair                                 │     │
│   │     ├── Context adaptive compression                         │     │
│   │     ├── Rate limit recovery (exponential backoff)            │     │
│   │     ├── Auth failure recovery (credential refresh)           │     │
│   │     └── Truncation recovery (rollback)                       │     │
│   │                                                               │     │
│   │  3. ACCUMULATE KNOWLEDGE                                      │     │
│   │     ├── Memory nudge → save declarative facts                │     │
│   │     ├── Skill nudge → save procedural knowledge              │     │
│   │     ├── Memory flush → pre-compression knowledge save        │     │
│   │     └── Session auto-saved to SQLite + JSON                  │     │
│   │                                                               │     │
│   │  4. DELEGATE & PARALLELIZE                                    │     │
│   │     ├── Subagent spawning for independent tasks              │     │
│   │     ├── Shared iteration budget (prevent runaway)            │     │
│   │     └── Summary-only results (context isolation)             │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  BETWEEN SESSIONS                                             │     │
│   │                                                               │     │
│   │  ├── MEMORY.md on disk (updated during session)              │     │
│   │  ├── USER.md on disk (updated during session)                │     │
│   │  ├── New skills in ~/.hermes/skills/                         │     │
│   │  ├── Session stored in state.db (FTS5 searchable)            │     │
│   │  ├── Honcho user model updated (async sync)                  │     │
│   │  └── Context length cache updated (if probed)                │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  SESSION N+1 (improved)                                       │     │
│   │                                                               │     │
│   │  ├── System prompt now includes:                             │     │
│   │  │   ├── Updated MEMORY.md (learned facts)                   │     │
│   │  │   ├── Updated USER.md (refined user model)                │     │
│   │  │   ├── New skills (reusable procedures)                    │     │
│   │  │   └── Better Honcho context (if enabled)                  │     │
│   │  │                                                            │     │
│   │  ├── Session search can recall Session N conversations       │     │
│   │  ├── Context limits cached (no re-probing)                   │     │
│   │  └── Skills available for matching to new tasks              │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  TRAINING-TIME (offline, periodic)                            │     │
│   │                                                               │     │
│   │  ├── Trajectory capture (save_trajectories=True)             │     │
│   │  ├── Batch runner (parallel trajectory generation)           │     │
│   │  ├── Trajectory compression (fit token budgets)              │     │
│   │  ├── RL training via Tinker-Atropos                          │     │
│   │  │   ├── Environment-based reward signals                    │     │
│   │  │   ├── GRPO/PPO optimization                               │     │
│   │  │   └── WandB metrics tracking                              │     │
│   │  └── Updated model weights deployed                          │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Feedback Signal Summary

| Signal | Type | Temporal Scope | Mechanism |
|--------|------|----------------|-----------|
| Tool call validation | Reactive | Intra-turn | Retry or recovery message |
| JSON repair | Reactive | Intra-turn | Auto-fix empty → `{}`, inject correction |
| Context compression | Adaptive | Intra-session | Summarize middle turns, preserve head/tail |
| Context probing | Adaptive | Cross-session | Step-down tiers, cache discovered limits |
| Memory nudge | Proactive | Intra-session | Periodic "[System: ...]" injection |
| Skill nudge | Proactive | Intra-session | After long tool loops |
| Memory flush | Proactive | Pre-compression | One-shot LLM call with memory tool only |
| Session search | Retrievative | Cross-session | FTS5 + LLM summarization |
| Honcho | External | Cross-session | AI-native user modeling service |
| RL training | Offline | Cross-deployment | GRPO/PPO on collected trajectories |
| Trajectory compression | Offline | Pre-training | Fit trajectories to token budgets |

---

## 9. Termination & Safety Bounds

| Bound | Default | Purpose |
|-------|---------|---------|
| `max_iterations` | 90 | Per-conversation LLM call limit |
| `IterationBudget` | 90 (shared) | Total across parent + all children |
| `MAX_CONCURRENT_CHILDREN` | 3 | Parallel subagent limit |
| `MAX_DEPTH` | 2 | No recursive delegation (child → grandchild rejected) |
| `max_retries` (API) | 6 | API error retry limit |
| `max_compression_attempts` | 3 | Context compression retry limit |
| Invalid tool retries | 3 | Hallucinated tool name retry limit |
| Invalid JSON retries | 3 | Malformed arguments retry limit |
| Codex incomplete retries | 3 | Continuation attempt limit |
| Memory char limits | 2200/1375 | Bounded knowledge store |

---

## Key Files

| File | Role in Self-Improving Loop |
|------|----------------------------|
| `run_agent.py` | Core loop, error recovery, nudge injection, compression orchestration |
| `agent/context_compressor.py` | Summarization, tool pair sanitization, boundary alignment |
| `agent/model_metadata.py` | Context probing tiers, limit parsing, token estimation |
| `tools/memory_tool.py` | Declarative knowledge persistence |
| `tools/skill_manager_tool.py` | Procedural knowledge persistence |
| `tools/todo_tool.py` | In-session task decomposition |
| `tools/delegate_tool.py` | Parallel subagent execution |
| `tools/session_search_tool.py` | Cross-session recall |
| `tools/rl_training_tool.py` | RL training interface (Tinker-Atropos) |
| `trajectory_compressor.py` | Post-hoc trajectory compression for training |
| `batch_runner.py` | Parallel trajectory generation |
| `hermes_state.py` | SQLite session persistence + FTS5 |
| `agent/prompt_builder.py` | System prompt assembly with skills + memory |
