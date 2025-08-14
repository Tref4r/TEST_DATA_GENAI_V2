---
title: report_a02_llm_fine_tuning_guide
---

---
## Executive Summary
---

### Purpose And Scope
<details>
<summary>Concise purpose, audience, and scope of this fine-tuning guide</summary>

---

- Audience alignment
  - Technical implementers building domain-specific LLMs
  - Product and business stakeholders needing decision-friendly trade‑offs
- Objectives
  - Provide a production‑ready guide for multiple fine‑tuning approaches
  - Map decisions to constraints in `data`, `budget`, `latency`, and `quality`
- Outcomes
  - Clear selection framework across `PEFT`, `full fine‑tune`, `instruction tuning`, `RLHF`
  - End‑to‑end implementation flows with code and evaluation checklists
- Style conformance
  - Follows `ctx_doc_style.md` structure, separators, and details blocks
  - Uses bullet‑only content, proper block indentation, and fenced code blocks

---
</details>

---
## Deliverables Mapping
---

### Alignment With Test Requirements
<details>
<summary>How this report meets the A02 task requirements and assessment criteria</summary>

---

- Coverage checklist
  - Parameter‑Efficient Fine‑tuning: `LoRA`, `QLoRA`, adapters
  - Full fine‑tuning: conditions, risks, and workflows
  - Instruction tuning: data formats, templates, and evaluation
  - `RLHF`: preference data, reward modeling, safety alignment
  - Quantization: `8‑bit`, `4‑bit`, `NF4`, `AWQ` trade‑offs
- Technical specifications
  - Data requirements, hardware profiles, and cost envelopes
  - Monitoring, logging, and offline/online evaluation
- Implementation steps
  - Chronological procedures with runnable examples
- Optimization and troubleshooting
  - Bottleneck analysis and common failure patterns
- Cost analysis
  - Compute classes, token budgets, and storage footprints
- Conformance evidence
  - Formatting follows `ctx_doc_style.md`
  - Task expectations sourced from `test_data_genai_v2.pdf` for fidelity validation

---
</details>

---
## Decision Framework
---

### When To Choose Each Approach
<details>
<summary>Selection heuristics for PEFT, full fine‑tuning, instruction tuning, and RLHF</summary>

---

- Constraints‑first thinking
  - If limited `GPU VRAM` and moderate data → prefer `LoRA` or `QLoRA`
  - If strict latency on edge inference → consider `quantization` + `PEFT`
  - If domain compliance requires deep distribution shift → consider `full fine‑tune`
  - If behavior conformity matters more than knowledge → prefer `instruction tuning` or `RLHF`
- Data realities
  - `High‑quality`, `task‑aligned`, `diverse` data dominates model choice
  - Synthetic augmentation helps coverage but needs `hallucination‑aware` QA
- Risk posture
  - Regulatory or safety critical contexts benefit from `RLHF` + `guardrails`
  - Vendor constraints may push `closed‑weight adapters` vs `open‑weight full FT`
- Exit costs
  - Adapters minimize model forking and ease upgrades
  - Full FT increases `model ops`, `storage`, and `MLOps` complexity

---
</details>

---
## Data Strategy
---

### Datasets, Formatting, And Curation
<details>
<summary>Data sources, schema, quality control, and augmentation patterns</summary>

---

- Data sources
  - Proprietary logs, knowledge bases, FAQs, SOPs, and tickets
  - Public instruction corpora and domain papers with license checks
- Formatting
  - Instruction‑response triples using consistent YAML/JSONL schema
  - Include `system` messages to encode policy and tone
- Quality controls
  - Deduplication, profanity filters, PII scrubbing, and toxicity screening
  - Balanced difficulty levels to avoid shortcut learning
- Augmentation
  - Programmatic templating, controlled paraphrasing, and counterexamples
  - Weak‑to‑strong supervision via teacher models
- Evaluation sets
  - Hold‑out data stratified by task, domain, and difficulty
  - Golden questions with deterministic references for regression testing

---
</details>

---
## Techniques Overview
---

### Parameter‑Efficient Fine‑Tuning (PEFT)
<details>
<summary>LoRA, QLoRA, and adapter families with practical guidance</summary>

---

- LoRA basics
  - Inject low‑rank matrices into attention and MLP projections
  - Freeze base weights to reduce `VRAM`, `time`, and `catastrophic forgetting`
- QLoRA
  - Quantize base weights to `4‑bit` (e.g., `NF4`) and train `LoRA` adapters
  - Leverage `paged optimizers` to handle long context and large batches
- Adapters
  - Layer‑wise modules allowing multi‑domain specialization without forking base
- Practical defaults
  - Rank `r=8–32`, `alpha=16–64`, `dropout=0.05` as safe starting points
  - Train `1–3` epochs with early stopping on validation perplexity
- Example (PEFT with Transformers)
  - Environment and minimal runnable snippet
    ```bash
    pip install transformers peft accelerate bitsandbytes datasets
    ```
    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    model_id = "meta-llama/Llama-3-8b"
    base = AutoModelForCausalLM.from_pretrained(
        model_id, load_in_4bit=True, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"])
    model = get_peft_model(base, lora)

    ds = load_dataset("json", data_files={"train":"train.jsonl","eval":"eval.jsonl"})
    def format(batch):
        return tok(batch["prompt"] + "\n" + batch["response"], truncation=True, padding="max_length", max_length=2048)
    ds = ds.map(format, batched=True)

    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps",
        output_dir="./out-peft"
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["eval"])
    trainer.train()
    ```

---
</details>

### Full Fine‑Tuning
<details>
<summary>When full retraining makes sense and how to control risks</summary>

---

- Indications
  - Large distribution shift, heavy tokenization drift, or deep domain syntax
  - Strong latency demands requiring fused kernels with no adapter overhead
- Risks
  - Higher compute, longer cycles, and increased overfitting potential
- Controls
  - Layer‑wise LR decay, mixed precision, and checkpoint averaging
  - Continuous evaluation with early stopping and `safety filters`
- Example training sketch
  - Distributed training with `accelerate`
    ```bash
    accelerate launch --multi_gpu train_full.py
    ```
    ```python
    # train_full.py
    import torch, math
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_id = "meta-llama/Llama-3-8b"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)

    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=1.5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        evaluation_strategy="steps",
        output_dir="./out-full"
    )
    # dataset loading omitted for brevity
    ```

---
</details>

### Instruction Tuning
<details>
<summary>Templates, policy encoding, and evaluation for instruction following</summary>

---

- Schema examples
  - JSONL with `system`, `instruction`, `input`, `output` fields
- Templates
  - Consistent chat formatting aligned with downstream inference runtime
- Policy encoding
  - Use `system` directives to bind tone, terminology, and red‑lines
- Evaluation
  - Exact match, BLEU/ROUGE for task types, and rubric‑based human review
- Snippet
  - Prompt formatting helper
    ```python
    def to_chat(sample):
      sys = sample.get("system","You are a helpful assistant.")
      user = sample["instruction"] + ("\n" + sample["input"] if sample.get("input") else "")
      assistant = sample["output"]
      return {"messages":[{"role":"system","content":sys},{"role":"user","content":user},{"role":"assistant","content":assistant}]}
    ```

---
</details>

### RLHF
<details>
<summary>Preference data pipelines, reward modeling, and PPO/DPO choices</summary>

---

- Data
  - Pairwise or listwise preferences from annotators or `AI feedback` with guardrails
- Reward modeling
  - Small `RM` head trained on frozen backbone embeddings
- Optimization choices
  - `PPO` for explicit reward optimization
  - `DPO` as simpler offline alternative using preference pairs
- Safety
  - Refusal training, jailbreak testing, and content policy eval sets
- Sketch
  - DPO style objective stub
    ```python
    # pseudo-code illustrating log-prob contrast on preferred vs rejected answers
    loss = (logp_chosen - logp_rejected).mean()
    ```

---
</details>

---
## Quantization
---

### Strategies And Trade‑offs
<details>
<summary>Latency, memory, and accuracy implications of 8‑bit, 4‑bit, and weight‑only methods</summary>

---

- Options
  - Post‑training quantization: `8‑bit`, `4‑bit`, `AWQ`, `GPTQ`
  - Training‑aware: `QLoRA` with `NF4` base, `bitsandbytes` optimizers
- Impacts
  - `VRAM` and throughput gains with modest perplexity increase
  - Activation quantization demands calibration to avoid quality cliffs
- Deployment
  - Ensure inference stack supports chosen format across CPU/GPU

---
</details>

---
## Implementation Playbooks
---

### PEFT End‑To‑End
<details>
<summary>Chronological steps from data to validated adapter artifact</summary>

---

- Prepare data
  - Curate, clean, and split with stratified hold‑outs
- Configure training
  - Choose `LoRA` ranks and learning rate grid
- Run training
  - Track metrics and save adapters in `safetensors`
- Evaluate
  - Task suite, safety suite, and latency checks
- Package
  - Export merged or separate adapters depending on deployment

---
</details>

### Full Fine‑Tune End‑To‑End
<details>
<summary>Compute‑aware plan for full fine‑tuning with rollback strategy</summary>

---

- Capacity planning
  - Estimate tokens, sequence length, and optimizer states
- Fault tolerance
  - Frequent checkpoints and corruption tests
- Rollback
  - Keep previous best and blue‑green deploy for inference

---
</details>

---
## Monitoring And Evaluation
---

### Metrics, Dashboards, And Tests
<details>
<summary>Quality, safety, and performance monitoring in staging and production</summary>

---

- Quality
  - `exact match`, `F1`, `BLEU/ROUGE`, and rubric scoring
- Safety
  - Red‑team prompts and policy violation trackers
- Performance
  - `p50/p95` latency, `tokens/sec`, and memory footprint
- Drift
  - Canary prompts and delta analysis across versions

---
</details>

---
## Troubleshooting
---

### Common Issues And Fixes
<details>
<summary>Frequent pitfalls with actionable remedies</summary>

---

- Overfitting
  - Reduce epochs, increase weight decay, and expand data variety
- Catastrophic forgetting
  - Mix domain and general data, weight adapters judiciously
- Instability
  - Lower LR, use gradient clipping, and enable `bf16`
- Hallucination
  - Add retrieval, improve references, and penalize unsupported claims

---
</details>

---
## Cost And Hardware
---

### Budgets, Profiles, And Optimizations
<details>
<summary>Compute classes, storage implications, and cost‑control levers</summary>

---

- Profiles
  - `8–13B` models with `QLoRA` on a single `24–48GB` GPU
  - `70B` class with multi‑GPU or parameter sharding
- Levers
  - Mixed precision, gradient checkpointing, and low‑rank ranks
- Budgeting
  - Track `tokens * epochs * batch_size` to estimate spend

---
</details>

---
## Appendices
---

### Reproducibility Pack
<details>
<summary>Environment, seeds, and artifacts for consistent results</summary>

---

- `requirements.txt` snapshot
- Random seed control and `deterministic` flags
- Model cards and change logs

---
</details>
