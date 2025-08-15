---
title: report_a02_prompt
---

---
## GenAI Workflow Prompts (Research, Planning, Implementation)
---

### Research Stage (Tools: Cursor, Claude, Windsurf)
<details>
<summary>Prompts for surveying techniques and extracting requirements</summary>

---

- **Cursor – requirement extraction**
  - "Scan `test_data_genai_v2.pdf` for A02 requirements. Return bullets: deliverables, evaluation criteria, expected line/file counts, visualization needs."
- **Claude – literature synthesis**
  - "Summarize `LoRA` vs `QLoRA` vs `full fine-tuning` under `VRAM`, `data size`, and `latency` constraints. Provide a decision table (bullets only)."
- **Windsurf – code scaffold**
  - "Generate a minimal `Transformers + PEFT` script for instruction JSONL with `LoRA r=16`, `alpha=32`, `dropout=0.05`; include an evaluation loop stub."
- **GPT – standards recall (ctx_doc_style)**
  - "List the strict formatting rules from `ctx_doc_style.md` as actionable checks I must pass (one bullet per rule) and show a short example for each."

---

</details>

### Planning Stage (Tools: GPT, Cursor)
<details>
<summary>Prompts for architecture, API, cost modeling, and quantization choices</summary>

---

- **GPT – architecture & integration**
  - "Propose end-to-end architecture for `PEFT + RLHF` with a model registry, blue–green deployment, and monitoring. Output a Mermaid diagram plus bullets for each component’s inputs/outputs and integration contracts."
- **Cursor – cost modeling**
  - "Given `tokens_per_sample`, `n_samples`, `epochs`, `tps_per_gpu`, `num_gpus`, `utilization`, and `$ / GPU_hour`, compute training cost and list sensitivity drivers. Emit a Python snippet with a function and example call."
- **GPT – quantization decision**
  - "Compare `4-bit nf4 (QLoRA)`, `8-bit`, and `bf16/fp16` for our target model size (7B/13B). Output bullets for VRAM, throughput implications, and quality risk; finish with a recommendation and fallback plan."

---

</details>

### Implementation Stage (Tools: GPT, Claude)
<details>
<summary>Prompts for runnable code, evaluation, rollout, and safety</summary>

---

- **GPT – runnable LoRA/QLoRA code**
  - "Produce a single-GPU SFT trainer using `Transformers + PEFT + TRL`, LoRA config `r=16, alpha=32, dropout=0.05`, targeting `q_proj,k_proj,v_proj,o_proj`. Include a `QLoRA (4-bit nf4)` variant using `BitsAndBytesConfig` with `device_map="auto"`. Keep it minimal and runnable."
- **Claude – DeepSpeed/Accelerate wiring**
  - "Generate an `accelerate launch` command using `ds_zero2.json` suitable for PEFT training. Include comments on gradient checkpointing and packing."
- **GPT – evaluation & sign-off**
  - "Create an evaluation checklist: exact match/F1, ROUGE-L, BERTScore; safety violations; latency `p50/p95`; regression gates; launch criteria; rollback rules. Return bullets only."
- **Claude – rollout SOP (blue–green + canary)**
  - "Draft a blue–green + canary SOP with gates: start at `5%` traffic, step `5→25→50→100%` if SLOs hold; auto-rollback on `p95`/errors/safety violations; include paging and post-mortem template."

---

</details>

---
## Prompt Evidence & Logging (Provenance)
---

### Execution Log Template
<details>
<summary>Record of prompts, models, and outputs for auditability</summary>

---

- **JSON template**
  - Use this to log each prompt run.
    ```json
    {
      "trace_id": "tr_{{uuid}}",
      "timestamp": "{{iso8601}}",
      "stage": "research|planning|implementation",
      "tool": "cursor|windsurf|gpt|claude|other",
      "model": "gpt-4o|claude-3.x|...|version",
      "prompt": "<verbatim prompt used>",
      "inputs": { "files": ["test_data_genai_v2.pdf","ctx_doc_style.md"], "params": { } },
      "output_digest": "sha256:{{hash}}",
      "notes": "observations, follow-ups"
    }
    ```
- **Storage**
  - Save logs under `logs/a02_prompts/*.jsonl`; reference `trace_id` in report cross-refs.

---

</details>

---
## Style-Constrained Output Wrappers
---

### ctx_doc_style Compliance Prompts
<details>
<summary>Wrappers to force bullet-only content, details blocks, and 2-space indentation</summary>

---

- **Wrapper – details block**
  - Template
    ```text
    Output ONLY one ### subsection with a single <details>…</details>.
    Inside <details>, start with a line containing only --- and end with a line containing only ---.
    All content MUST be bullet points; no paragraphs. Indent all code/tables/Mermaid TWO spaces under a bullet.
    ```
- **Wrapper – table & code**
  - Template
    ```text
    Place each table/code/Mermaid under a parent bullet labeled "Table" or "Diagram" or "Example".
    ```

---

</details>

---
## Assignment Meta (Duration, Naming, Line Targets)
---

### Duration & Deliverables (for reviewer clarity)
<details>
<summary>Timebox, naming, files, and expected report length</summary>

---

- **Duration**
  - OUT-OF-OFFICE completion, total timebox: `5 days`.
- **Primary deliverables**
  - `report_<task>.md` (at least one main file per task).
- **Multiple main files**
  - Use `report_<task>_part01_<part_name>.md` naming.
- **Prompt file naming**
  - `report_<task>_prompt.md` or `report_<task>_part_prompt.md` (this file).
- **Focus**
  - Framework analysis, system design, technology evaluation.
- **Report specs**
  - `~1000–1500` lines per report file, `2–10` files per task.
- **Audience**
  - Multi-audience accessibility: technical + business stakeholders.

---

</details>

---
## Cross-References & Traceability
---

### Where each A02 deliverable is satisfied
<details>
<summary>Link prompts to the corresponding report sections</summary>

---

- **Architecture & integration** → `report_A02_part01_architecture.md` (system, specs, runnable LoRA/QLoRA).  
- **Cost & trade-offs** → `report_A02_part02_cost_benefit.md` (quantization impact, budgeting, decision helper).  
- **Implementation & rollout** → `report_A02_part03_impl_plan.md` (WBS, releases, rollback, sign-off).  
- **Monitoring & reliability** → `report_A02_part04_monitoring.md` (SLOs, logs/traces, alerts, drift).  
- **Workflow & prompts** → `report_A02_part05_workflow.md` (agent flows, APIs, error handling, observability).

---

</details>
