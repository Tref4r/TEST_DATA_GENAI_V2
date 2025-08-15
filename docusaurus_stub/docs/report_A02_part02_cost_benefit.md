---
title: report_a02_part02_cost_benefit
---

---
## Framework Comparison
---

### Fine-Tuning Approach Trade-offs
<details>
<summary>Concise matrix contrasting LoRA, QLoRA, full fine-tuning, instruction tuning, and RLHF</summary>

---

- **Comparison axes**
  - Cost, data requirement, quality ceiling, complexity, VRAM need, latency impact.

- **Summary table**
  - Compact view of differences (keywords only).

    | **Aspect** | LoRA | QLoRA | Full FT | Instruction | RLHF |
    |---|---|---|---|---|---|
    | **VRAM** | Low | Very low | High | Medium | High |
    | **Data** | Low–Med | Low–Med | High | Medium | High |
    | **Latency** | Low | Low | Lowest | Medium | Medium |
    | **Quality** | Medium | Med–High | Highest | High | Highest |
    | **Complexity** | Low | Medium | High | Medium | Very high |

- **Interpretation**
  - **PEFT** fits constrained budgets and fast iteration.
  - **Full fine-tuning** suits deep distribution shifts at higher cost.
  - **RLHF** adds alignment quality with significant pipeline complexity.

---

</details>

---
## Cost–Benefit & Trade-offs
---

### Compute Budgeting, Integration Complexity, And Exit Costs
<details>
<summary>Quantitative heuristics and selection guidance</summary>

---

- **Cost and complexity matrix**
  - Quick reference by `budget`, `latency`, and `quality` constraints.

    | **Dimension** | **LoRA** | **QLoRA** | **Full FT** | **Instruction** | **RLHF** |
    |---|---|---|---|---|---|
    | **OpEx** | Low | Very low | High | Mid | Very high |
    | **Integration** | Low | Mid | High | Mid | Very high |
    | **Maintenance** | Low | Low | High | Mid | High |
    | **Exit cost** | Low | Low | High | Mid | High |

- **Budget modeling**
  - Tokens: `tokens_total = tokens_per_sample * n_samples * epochs`.
  - Cost: `≈ tokens_total / throughput * $/GPU_hour`.
  - Sensitivity drivers: sequence length, batch size, gradient accumulation, precision, and checkpointing cadence.

- **Hidden costs**
  - Data labeling, preference collection, eval set maintenance, on-call runbooks.

---

</details>

### Quantization & Hardware Impact on Cost
<details>
<summary>How 4-bit/8-bit and GPU sizing change total cost and feasibility</summary>

---

- **Quantization effects**
  - **4-bit (QLoRA / `nf4`)**: minimizes VRAM; enables larger base models on fewer GPUs; small overhead in compute vs savings in memory.
  - **8-bit**: more stable for larger effective batch sizes; still reduces memory footprint vs FP16.
  - **Full precision (fp16/bf16)**: best training stability/perf; highest VRAM/OpEx.

- **GPU sizing heuristics (guidance)**
  - **PEFT/QLoRA on 7B**: feasible on a single `24–48GB` GPU with gradient accumulation.
  - **Full FT 7B–13B**: plan multiple `24–48GB` GPUs with **ZeRO-2/3** and optional CPU/NVMe offload to control peak memory.
  - **Throughput drivers**: sequence length, batch size, optimizer, attention kernels, and checkpointing cadence.

- **Cost levers**
  - **Reduce VRAM** (4-bit/8-bit) → fewer/lower-tier GPUs → lower `$ / GPU_hour × hours`.
  - **Increase throughput** (flash-attention, packing, grad-checkpointing) → fewer training hours for the same tokens budget.

---

</details>

### Budget Calculator (Tokens → Hours → $)
<details>
<summary>Drop-in formula + reference Python to estimate training hours and cost</summary>

---

- **Formulas**
  - Tokens: `tokens_total = tokens_per_sample * n_samples * epochs`
  - Hours: `train_hours ≈ tokens_total / (tokens_per_second_per_gpu * num_gpus * utilization)`
  - Cost: `cost ≈ train_hours * $/GPU_hour * num_gpus`

- **Reference Python**
  ```python
  def estimate_cost(tokens_per_sample, n_samples, epochs,
                    tps_per_gpu, num_gpus, utilization,
                    price_per_gpu_hour):
      tokens_total = tokens_per_sample * n_samples * epochs
      hours = tokens_total / (tps_per_gpu * num_gpus * utilization)
      cost = hours * price_per_gpu_hour * num_gpus
      return dict(tokens_total=tokens_total, train_hours=hours, cost_usd=cost)

- **Usage notes**
  - Fill `tps_per_gpu`, `$ / GPU_hour`, and `utilization` with your actual cluster/cloud stats.
  - Sensitivity: sequence length and batch size heavily influence `tps_per_gpu`.

---

</details>

### Decision Helper (Cost–Quality–Latency)
<details>
<summary>Mermaid decision flow to pick a fine-tuning strategy</summary>

---

- **Flow**
  ```mermaid
  flowchart TD
    A[Budget tight / Low VRAM?] -->|Yes| B[QLoRA]
    A -->|No| C[Domain shift large?]
    C -->|Yes| D[Full FT + ZeRO-3]
    C -->|No| E[LoRA / Adapters]
    D --> F[Need alignment->DPO/RLHF]
    E --> F
    B --> F

- **Interpretation**
  - Start with **PEFT**; escalate to **Full FT** only for big domain shifts or strict control.
  - Add **DPO/RLHF** when preference alignment is a hard requirement.

---
</details>