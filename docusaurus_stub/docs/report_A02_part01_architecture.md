---
title: report_a02_part01_architecture
---

---
## GenAI Architecture & Integration Plan
---

### End-to-End Fine-Tuning System Architecture
<details>
<summary>System components, data flows, and integration patterns</summary>

---

- **Architecture goals**
  - Modular, scalable fine-tuning across multiple LLM sizes.
  - Unified orchestration supporting **PEFT**, **full fine-tuning**, **instruction tuning**, and **RLHF**.
  - Cost-aware scheduling across heterogeneous hardware profiles.

- **High-level components**
  - **Data layer**: ingest, cleaning, augmentation, evaluation-set curation.
  - **Training layer**: distributed orchestrator for GPU/TPU backends.
  - **Evaluation layer**: quality, safety, and performance benchmarks.
  - **Deployment layer**: packaging, versioning, and API serving.
  - **Monitoring layer**: continuous evaluation, drift detection, and safe rollback.

- **Mermaid – architecture**
  - Diagram
    ```mermaid
    graph TD
      A[Data Sources] --> B[Data Processing Pipeline]
      B --> C[Training Orchestrator]
      C --> D[Model Registry]
      D --> E[Deployment Service]
      E --> F[Inference API]
      F --> G[Monitoring & Logging]
      G --> C
    ```

- **Integration patterns**
  - API-driven orchestration using Airflow, Prefect, or LangGraph.
  - Shared embedding store to enable retrieval-augmented evaluation.
  - Blue–green deployment with shadow testing prior to promotion.

---

</details>

### Integration Specifications
<details>
<summary>REST/streaming APIs, batch jobs, and fallback/error handling</summary>

---

- **Inference API (REST)**
  - Endpoints: `POST /v1/generate`, `POST /v1/score`, `GET /v1/models`.
  - Contracts: `trace_id`, `session_id`, `user_id`; response fields include `latency_ms`, `tokens_used`.

- **Streaming**
  - Server-sent events or WebSocket with heartbeat and backpressure.

- **Real-time vs batch**
  - Real-time: dynamic batching and prefill caching; SLA `p95 < 1200ms`.
  - Batch: nightly evaluation and offline scoring runs.

- **Fallback and error handling**
  - Tiered policy: `adapter → base → safe_template`.
  - Circuit breaker, exponential backoff with jitter, and dead-letter queues.

- **SLOs and limits**
  - Enforcement: `p95 < 1200ms` triggers automatic rollback on violation.
  - Per-key rate limits and budget guardrails for cost control.

---

</details>

### Terminology Standardization
<details>
<summary>Consistent terms used across all A02 parts</summary>

---

- **PEFT**: parameter-efficient fine-tuning (adapters/LoRA/QLoRA) — train small modules, freeze base.
- **LoRA**: low-rank adapters injected into linear layers; freezes base weights; reduces trainable params. 
- **QLoRA**: 4-bit quantized base (`nf4`, double-quant) + LoRA adapters, preserving quality with low VRAM.
- **Instruction Tuning (SFT)**: supervised fine-tuning on `(instruction, input, output)` pairs.
- **RLHF / DPO**: preference-based post-training (pairwise `chosen/rejected`); DPO optimizes directly from preferences.
- **ZeRO**: partition optimizer/grad/params across devices; offload to CPU/NVMe for memory scaling.
- **SLO/SLA**: target vs contractual latency/availability; we track `p50/p95` and rollback on violation.

---

</details>

### Fine-Tuning Strategy Comparison (Use-Case Oriented)
<details>
<summary>When to pick LoRA/QLoRA/Adapters vs Full FT vs Instruction vs RLHF</summary>

---

- **PEFT (LoRA/Adapters)** 
  - **Use when**: tight budget/VRAM, fast iteration, many domain variants (swap adapters).
  - **Notes**: freeze base; small trainable params; often `target_modules="all-linear"` for coverage.
- **QLoRA (4-bit)** 
  - **Use when**: very limited VRAM; aim to retain near full-precision quality with `nf4` quantization.
  - **Notes**: double-quant + paged optimizers reduce memory spikes; enables very large bases on single high-VRAM GPUs.
- **Full fine-tuning** 
  - **Use when**: large domain shift or deep control of behavior is required.
  - **Notes**: higher cost; plan ZeRO-2/3 + (optional) offload to scale.
- **Instruction tuning (SFT)** 
  - **Use when**: you need “follow-instruction” behavior; pair well with PEFT/QLoRA to save cost.
- **RLHF / DPO**
  - **Use when**: you must align outputs to human preferences/safety; DPO is simpler/more stable than PPO in many cases.

---

</details>

### Minimal Runnable Example (PEFT LoRA Quickstart)
<details>
<summary>Supervised fine-tuning with LoRA adapters (Transformers + PEFT + TRL)</summary>

---

- **Environment**
  - `pip install transformers peft accelerate trl datasets bitsandbytes`

- **Quickstart script (single GPU)**
  ```python
  from datasets import load_dataset
  from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
  from peft import LoraConfig
  from trl import SFTTrainer

  base = "mistralai/Mistral-7B-v0.1"
  tok = AutoTokenizer.from_pretrained(base, use_fast=True)
  tok.pad_token = tok.eos_token

  model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")

  peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj","k_proj","v_proj","o_proj"])

  data = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

  args = TrainingArguments(
    output_dir="out/lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    bf16=True
  )

  trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=peft,
    train_dataset=data,
    max_seq_length=1024,
    packing=True,
    args=args
  )
  trainer.train()
  trainer.model.save_pretrained("out/lora_adapter")

- **QLoRA variant (4-bit `nf4`)**
  ```python
  from transformers import BitsAndBytesConfig, AutoModelForCausalLM

  quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
  )

  model = AutoModelForCausalLM.from_pretrained(
    base,
    quantization_config=quant,
    device_map="auto"
  )

- **Notes**
  - Start with **LoRA**; switch to **QLoRA** for tight VRAM budgets (`nf4`, `target_modules="all-linear"` often helps).
  - For **full FT**, plan **ZeRO-2/3** and consider offload if you hit memory limits.

---

</details>

### Cross-References & Traceability
<details>
<summary>Where each A02 deliverable is covered across files</summary>

---

- **Strategy comparison & cost trade-offs** → `report_A02_part02_cost_benefit.md`
- **Implementation plan & milestones** → `report_A02_part03_impl_plan.md`
- **Monitoring, performance & reliability** → `report_A02_part04_monitoring.md`
- **Workflow, prompts & agent orchestration** → `report_A02_part05_workflow.md`
- **GenAI workflow prompts (proof of AI-assisted process)** → `report_A02_part01_prompt.md`

---

</details>
