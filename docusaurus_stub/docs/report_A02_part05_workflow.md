---
title: report_a02_part05_workflow
---

---
## GenAI Workflow Design
---

### Prompt Engineering, Model Interaction, And Agent Coordination
<details>
<summary>Prompt patterns, routing strategies, and multi-agent collaboration</summary>

---

- **Prompt engineering**
  - Instruction-first prompts, style adapters, anti-hallucination guards, grounding to context.

- **Model interaction**
  - Multi-turn strategies for complex tasks, self-check prompts, adapter-aware routing.

- **Agent coordination**
  - Planner → Worker → Reviewer with JSON contracts and deterministic hand-offs.

- **Mermaid – agent flow**
  - Diagram
    ```mermaid
    graph TD
      Planner["Planner"] --> Worker["Worker"]
      Worker --> Reviewer["Reviewer"]
      Reviewer -->|approve| Worker
      Worker --> Registry["Registry"]
    ```

---

</details>

### API & Tool Invocation Schema
<details>
<summary>Function/tool call contracts and external API specs</summary>

---

- **Function/Tool call contract (LLM → tool)**
  - JSON schema (contract)
    ```json
    {
      "type": "object",
      "properties": {
        "tool": { "type": "string" },
        "args": { "type": "object" },
        "trace_id": { "type": "string" }
      },
      "required": ["tool", "args"]
    }
    ```
  - Example message (LLM asks to call a tool)
    ```json
    {
      "tool": "search_knowledge_base",
      "args": { "query": "refund policy for premium tier" },
      "trace_id": "tr_9f12"
    }
    ```
- **External API (service boundary)**
  - Table
    | **Endpoint** | **Method** | **Body** | **Notes** |
    |---|---|---|---|
    | `/v1/agents/plan` | POST | `messages[]` | returns plan JSON with steps |
    | `/v1/tools/exec` | POST | `{ tool, args }` | executes a tool with args |
    | `/v1/review` | POST | `{ draft, criteria[] }` | returns review verdict |
- **Design notes**
  - Always attach `trace_id` for cross-service correlation (ties to Part 4 logging schema).
  - Keep JSON contracts versioned (e.g., `x-contract-version` header).

---

</details>

### Error Handling & Fallback Inside Agents
<details>
<summary>Retries, timeouts, circuit breakers, and human escalation</summary>

---

- **Policies**
  - **Timeouts**: per step `<= 10s` (tool), `<= 2s` (cache), `<= 60s` (LLM).
  - **Retries**: idempotent tools only (`max=2`, backoff `500ms, 2s`).
  - **Circuit breaker**: open after `p95` or error-rate breach for `> 3m`; route to fallback.
- **Fallbacks**
  - **Cache/gist**: return last good answer with disclaimer if upstream is degraded.
  - **Human-in-the-loop**: escalate when confidence `< threshold` or safety flags present.
- **Reference snippet**
  ```python
  import time

  class TransientError(Exception):
      pass

  def call_tool(exec, args, timeout_s=10):
      for backoff in [0.5, 2.0]:
          try:
              return exec(args, timeout=timeout_s)
          except TransientError:
              time.sleep(backoff)
      raise

---      

</details>

### Real-Time vs Batch Orchestration
<details>
<summary>Low-latency path vs scheduled processing patterns</summary>

---

- **Real-time (sync)**
  - Flow: Request → Router → Planner → Tools → Draft → Reviewer → Response (`p95` budget).

- **Batch (async)**
  - Flow: Schedule → Planner → Tools → Draft → Reviewer → Publish/Notify.

- **Mermaid – orchestration overview**
  - Diagram
    ```mermaid
    flowchart LR
      In["Request/Schedule"] --> R["Router/Planner"]
      R --> T["Tools/Functions"]
      T --> D["Draft"]
      D --> V["Reviewer"]
      V --> Out["Response/Publish"]
    ```

- **Selection criteria**
  - Use real-time for interactive UX; batch for large document sets or periodic data jobs.

---

</details>

### Prompt Pattern Library 
<details>
<summary>Reusable snippets for grounding, JSON mode, and self-check</summary>

---

- **Grounded answer (with sources)**
  - Template
    ```text
    You must answer strictly using the provided context.
    If the context is insufficient, say "I don't have enough information."
    Format:
    - Answer: <concise answer>
    - Sources: <list of source ids>
    ```

- **JSON mode (strict schema)**
  - Template
    ```text
    Return ONLY valid JSON that matches this schema:
    { "decision": "approve|reject", "reasons": [string], "confidence": 0.0-1.0 }
    ```

- **Self-check pass**
  - Template
    ```text
    Before finalizing, list 3 potential errors in your draft and fix them if found.
    Output only the corrected draft.
    ```

---

</details>

### Observability Hooks (tie to Part 4)
<details>
<summary>Emit traces, counters, and cost metrics from each agent step</summary>

---

- **Per-step emissions**
  - `trace_id`, `span_id`, `model_id/adapter_id/version`, `input_len`, `output_len`, `latency_ms`, `safety_flags`, `cost_usd`.

- **Minimal Python stub**
  ```python
  def emit(step, **kvs):
      # plug into your metrics/trace client
      pass

  # Example:
  emit("planner", trace_id="tr_123", latency_ms=123, input_len=512, output_len=128, cost_usd=0.0012)

- **Cross-refs**
  - Logging/Tracing fields map to Part 4 “Logging & Tracing Schema” and Alert rules.
  
---

</details>

---
## Stakeholder Materials (Extended)
---

### Executive One-Pager And Developer Runbook
<details>
<summary>Audience-specific materials for business and engineering teams</summary>

---

- **Executive one-pager**
  - ROI, risk register, KPIs, and roadmap themes.

- **Developer runbook**
  - SOPs, templates, and YAML samples for release checklists and evaluations.

---

</details>
