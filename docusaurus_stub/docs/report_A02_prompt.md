---
title: report_a02_prompt
---

---
## Strategic GenAI Workflow Prompts
---

### Research And Planning
<details>
<summary>Prompts used to plan the fine‑tuning guide and compare approaches</summary>

---

- "Synthesize a decision matrix contrasting `LoRA`, `QLoRA`, `full FT`, `instruction tuning`, and `RLHF` under `budget`, `data`, and `latency` constraints. Return bullets only."
- "List failure modes for QLoRA on long‑context tasks and propose mitigations. Focus on concrete hyperparameters and ablations."
- "Draft a PEFT training runbook with reproducible commands and checkpoints for `8B` and `70B` models."

---
</details>

### Implementation Assistance
<details>
<summary>Prompts used to generate runnable code and evaluation suites</summary>

---

- "Write a minimal `Transformers + PEFT` training script for JSONL instruction data. Include tokenization, `LoRA` config, and `Trainer` setup."
- "Provide evaluation snippets for exact match and rubric‑based scoring. Use clear, testable helpers with docstrings."
- "Suggest telemetry fields for latency and token throughput suitable for a production dashboard."

---
</details>
