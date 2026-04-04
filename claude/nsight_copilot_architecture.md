# Nsight Copilot Architecture

## Overview

Nsight Copilot is an AI-powered assistant embedded in NVIDIA Nsight Compute (NCU). It ships as a Python wheel (`nsight_copilot-2026.1.0`) found in the NCU install directory (`extras/python/`). It is built on **LangChain + LangGraph** and provides two interfaces:

1. **gRPC server mode** — NCU-UI launches the Python backend automatically; the chat window in NCU-UI communicates via protobuf RPCs
2. **TUI mode** — terminal-based interface using the `textual` library

The core architecture is a **LangGraph `StateGraph`** with conditional routing, tool-calling LLMs, and a Reflective RAG agent for documentation retrieval.

---

## High-Level Data Flow

```
User question (+ optional .ncu-rep report path + result_id)
        │
        ▼
   ┌─────────┐
   │ Chatbot  │  (chatbot.py)
   │  Graph   │
   └────┬─────┘
        │
   ┌────┴────────────────────┐
   │ init → remove_old_msgs  │
   │          │               │
   │    has_report?           │
   │   ┌──yes──┴──no──┐      │
   │   ▼              ▼      │
   │ Report        Non-Report │
   │ Subgraph      Subgraph   │
   │   │              │      │
   │   └──────┬───────┘      │
   │          ▼               │
   │        END               │
   └──────────────────────────┘
```

The outer graph (`ChatbotGraphBuilder`) routes based on whether a report path is provided:

| Node | What it does |
|------|-------------|
| `init` | Extracts `user_prompt` from last message, initializes `docs`/`doc_references` |
| `remove_old_messages` | Prunes cancelled queries, old retrieval context, and tool call artifacts from history |
| `_has_report` | Conditional edge: `"yes"` → report subgraph, `"no"` → non-report subgraph |

---

## Two Subgraphs

### 1. Non-Report Subgraph (general questions)

**File**: `chatbot.py:155` — `NonReportSpecificSubgraphBuilder`

```
START → retrieve_docs → answer_using_docs ⇄ tools → END
```

- **`retrieve_docs`**: Invokes `DocsRetriever` (wrapping the `ReflectiveRAGAgent`) to fetch relevant documentation chunks from NCU, CUDA, and/or OptiX docs
- **`answer_using_docs`**: Calls `DocumentationRagBuilder` chain — system prompt + retrieved docs + LLM w/ tools
- **`tools`**: LangGraph `ToolNode` — if the LLM decides to call a tool (e.g., `documentation_retriever` for additional docs), the output loops back to `answer_using_docs`

Uses the **`DocumentationRagBuilder`** chain (`models.py:204`):
```
ChatPromptTemplate → log_prompt → truncate_to_context_window → LLM_with_tools
```

### 2. Report Subgraph (profile-result questions)

**File**: `chatbot.py:268` — `ReportSpecificSubgraphBuilder`

```
START → retrieve_details_page → answer_using_report_pages ⇄ tools → END
```

- **`retrieve_details_page`**: Unconditionally extracts the **Details Page** from the NCU report and injects it as a `HumanMessage`. This includes sections, header metrics, and rule results
- **`answer_using_report_pages`**: Calls `ReportRagBuilder` chain with the Details Page context
- **`tools`**: Two tools available:
  - `source_page_tool` — lazily retrieves the **Source Page** (source-correlated per-PC metrics) on demand
  - `DocsRetrieverTool` — searches NCU/CUDA/OptiX documentation

Uses the **`ReportRagBuilder`** chain (`models.py:259`), identical structure to `DocumentationRagBuilder` but with a different system prompt context paragraph indicating a profile result is attached.

---

## How It Leverages NCU Under the Hood

Nsight Copilot uses the `ncu_report` Python API (SWIG bindings, also in `extras/python/`) to extract profiling data from `.ncu-rep` files. This happens through two `ReportPage` subclasses:

### DetailsPage (`report_pages/details_page.py`, 908 lines)

Extracts NCU's structured analysis sections:
- **Sections**: `IAction.sections()` → section name, description
- **Header metrics**: Per-section key metrics (e.g., compute throughput, memory throughput)
- **Body metrics**: Detailed metric breakdowns within each section
- **Rule results**: Automated analysis with severity, speedup estimates, focus metrics, and result tables
- Serialized via `DetailsPageTableSerializer` into tabulated text

The Details Page is always included in the LLM prompt when a report is attached.

### SourcePage (`report_pages/source_page.py`, 511 lines)

Extracts source-correlated metrics — the same data shown in NCU-UI's Source tab:

**Default metrics extracted:**
- `inst_executed` — per-PC instruction execution count
- `group:smsp__pcsamp_warp_stall_reasons` — all-cycle warp stall sampling
- `group:smsp__pcsamp_warp_stall_reasons_not_issued` — not-issued cycle stall sampling
- `memory_type` — address space (Global/Local/Shared/Texture/Tmem)
- `memory_access_type` — Load/Store/Atomic/Shift
- `memory_access_size_type` — access size in bits
- `derived__memory_l1_wavefronts_shared_excessive`
- `derived__memory_l2_theoretical_sectors_global_excessive`

**How it works:**
1. For each metric, `build_pc_to_metric_value_dict()` iterates over instanced metric values via `metric.correlation_ids().value(i)` → PC, `metric.value(i)` → value
2. Group metrics are expanded: `metric.value()` returns comma-separated sub-metric names, each is resolved and summed per-PC
3. Builds two data structures:
   - `ll_lines` (low-level): `dict[PC → LowLevelSourceLine]` with SASS instruction + metric values
   - `hl_lines` (high-level): `dict[file → dict[line → HighLevelSourceLine]]` via `action.source_info(pc)`, with aggregated metrics
4. Adds source markers (performance issue annotations) from `action.source_markers()`
5. Serialized via `SourcePageTableSerializer` into CUDA+SASS interleaved tables

The Source Page is **not** always included — it's retrieved on demand via `source_page_tool` when the LLM decides it needs source-level information. Truncated to 1/4 of max context window to avoid overflow.

### NcuReportCache (`utilities/ncu_report_cache.py`)

Thread-safe two-level cache (report-level + result-level) with mtime-based invalidation. The singleton `ncu_report_cache` avoids re-parsing `.ncu-rep` files across multiple page requests.

---

## Reflective RAG Agent

**File**: `agents/reflective_rag_agent.py` — `ReflectiveRAGAgent(Runnable)`

A LangGraph sub-pipeline used by `DocsRetriever` for documentation retrieval with quality control:

```
init_state → retrieve → grade → ─── sufficient? ───┬─ yes ─→ filter_and_rerank → END
                                    │                │
                                    │               no
                                    │                │
                                    │                ▼
                                    │             search (forum)
                                    │                │
                                    │                ▼
                                    │              grade ──── no_after_search ─→ END
                                    └────────────────┘
```

| Stage | Details |
|-------|---------|
| **retrieve** | Queries all configured retrievers (NCU docs, CUDA docs, OptiX docs) |
| **grade** | LLM-based relevance scoring: each doc gets a 1-5 score via a JSON grader prompt. Uses `BatchGradingStrategy` with configurable concurrency (default 10) |
| **sufficient?** | Needs ≥2 accepted docs (score ≥3) OR ≥1 good doc (score ≥4). If insufficient and search not yet attempted → search |
| **search** | Falls back to web/forum search (developer forums) |
| **filter_and_rerank** | Keeps docs with score ≥3, sorts by score descending |

---

## Tools

### DocsRetrieverTool (`tools.py:33`)
- LangChain `BaseTool` wrapping `DocsRetriever`
- Searches NCU, CUDA, and OptiX documentation
- Configurable: `use_ncu_docs`, `use_cuda_docs`, `use_optix_docs`, `use_forum_search`
- Returns `Command` to update graph state with retrieved docs and references
- Available in both subgraphs

### source_page_tool (`tools.py:144`)
- `@tool` decorated function
- Creates a `SourcePage` from the report in `state["report_path"]`
- Always requests CUDA source type with source markers enabled
- Truncates output to 1/4 of max context window
- Available only in the report subgraph

---

## LLM Configuration

**File**: `chatbot.py:578` — `ChatbotBuilder`

Two model tiers:
- **`large_reasoning_model`** — primary answering LLM (with fallback chain)
- **`medium_reasoning_model`** — helper LLM used for RAG grading and search query generation

Two provider backends:
- **`LLMProviderType.NVIDIA`** — `ChatNVIDIA` via `langchain_nvidia_ai_endpoints` (NVIDIA AI endpoints)
- **`LLMProviderType.OPENAI`** — `ChatOpenAI` for OpenAI-compatible endpoints

Both use a shared `InMemoryRateLimiter` and support streaming. Fallback chains via `llm.with_fallbacks()` handle provider failures gracefully.

---

## gRPC Server Interface

**File**: `server/servicer.py` — `ChatServicer(chat_pb2_grpc.ChatServicer)`

NCU-UI communicates with the Python backend via async gRPC:

| RPC | Purpose |
|-----|---------|
| `CreateChat` | Initialize a new chat session |
| `DeleteChat` | Clean up a chat session |
| `GetAvailableCredits` | Check API credit balance (currently `NotImplementedError`) |
| `CreateQuery` | Submit a user query with optional report context |
| `CancelQuery` | Cancel an in-progress query |
| `RetryLastQuery` | Re-run the previous query |
| `GetPartialQueryResult` | **Response-streaming RPC** — yields `PartialQueryResult` messages as LLM tokens arrive |

The servicer delegates to `ChatHistoryManager` which manages multiple concurrent chat sessions, each backed by a `Chatbot` instance with LangGraph's `InMemorySaver` checkpointer for conversation persistence.

---

## Prompt Engine

**File**: `prompt_engine/` — `PromptClassifier` + `PromptBuilderFactory`

Classifies user input into `PromptType` (e.g., `FREEFORM` vs structured commands like "explain"). For non-freeform prompts, a `PromptBuilder` transforms the user's input into an enhanced prompt template before passing it to the LLM.

---

## System Prompt

**File**: `models.py:15` — `NSIGHT_COPILOT_SYSTEM_PROMPT`

Key constraints baked into the identity:
- Professional NVIDIA assistant focused on NCU, CUDA profiling, OptiX
- Guidance over code generation — no full kernel rewrites
- No competitor comparisons (AMD/Intel)
- Concise, scannable markdown formatting
- Context-aware: different prompts depending on whether a profile result is attached

---

## Package Dependencies

From `METADATA`:
- **Core**: `langchain-core`, `langgraph`, `langchain-nvidia-ai-endpoints`, `langchain-openai`
- **Server**: `grpcio`, `protobuf`
- **TUI**: `textual`
- **Utilities**: `tabulate`, `pydantic`, `httpx`, `pyyaml`
- **Python**: ≥3.11

---

## Summary: Architecture in One Sentence

Nsight Copilot is a LangGraph chatbot that routes user questions through either a documentation-only RAG pipeline or a report-aware pipeline (injecting `DetailsPage` unconditionally + `SourcePage` on-demand via tool call), where both NCU report data extraction and documentation retrieval use LLM-graded relevance filtering before being presented to the answering LLM.
