---
title: Report B01 – Prompt Workflow
---

---
## Genai workflow and prompts
---

### Objectives and strategy
<details>
<summary>What the prompts aim to produce and how they are sequenced</summary>

---

- **Goals**
  - Produce a practical, implementation-first tutorial for vector databases with clear code and ops guidance.
  - Compare leading tools for stakeholder selection; deep dive one OSS tool (Qdrant) with runnable steps.
- **Constraints**
  - Follow `ctx_doc_style.md` strictly (bullets-only, separators, details blocks, indentation).
  - Align with B01 deliverables in `test_data_genai_v2.pdf` (concepts, comparison, deep dive, implementation, best practices).
- **Tooling**
  - Use a general-purpose LLM for drafting; verify commands and APIs against official docs; keep examples minimal and runnable.
- **Verification**
  - Cross-check quickstart commands, API methods, and hybrid search capabilities from official documentation before finalizing examples.

---

</details>

### Drafting prompts examples
<details>
<summary>Reusable prompt snippets (copy/paste) for future team use</summary>

---

- **Concept distillation**
  - “Explain vector DB basics for an engineering audience in plain language. Output as short bullets. Cover embeddings, indexes (`HNSW`, IVF), distance metrics (`cosine`, `dot`, `L2`), and typical use cases.”
- **Comparison matrix**
  - “Build a concise table comparing Qdrant, Weaviate, Milvus, Pinecone, and Chroma. Columns: hosting, hybrid support, indexes, filtering, APIs, ops. Keep claims conservative and verifiable.”
- **Grounded code generation**
  - “Write a Python snippet that creates a Qdrant collection (`size=384`, `Distance.COSINE`), upserts 3 points with payload, and runs a filtered `query_points` search. Use only `qdrant-client` public APIs from the latest quickstart.”
- **Hybrid fusion example**
  - “Show Qdrant `RRF` fusion with `prefetch` over sparse and dense named vectors using the Python client. Keep it minimal.”
- **Ops checklist**
  - “List pragmatic ops tasks for Qdrant in production: snapshots, version pinning, metrics, capacity planning, auth/TLS.”

---

</details>

### Review and style prompts
<details>
<summary>Style guardrails and automated checks via LLM</summary>

---

- **Structure check**
  - “Validate that every `##` has only `###` children wrapped in a single `<details>` block, and separators are placed per style rules.”
- **Block indentation check**
  - “Inspect code blocks and mermaid graphs; ensure they’re indented exactly 2 spaces under their parent bullet.”
- **Audience pass**
  - “Rewrite any overly technical sentence to plain language while keeping correctness.”

---

</details>

### Prompt log
<details>
<summary>Sequence of actual prompts used and outcomes (evidence of GenAI utilization)</summary>

---

- Initial scoping
  - Prompt:
    ```text
    Summarize B01 task requirements from the test PDF. Output concise bullets listing all deliverables.
    ```
  - Outcome:
    - Checklist of B01 deliverables mapped to sections.

- Concept drafting
  - Prompt:
    ```text
    Draft bullets explaining embeddings, distance metrics (cosine/dot/L2), HNSW/IVF, and typical vector DB use cases.
    Keep language plain and concise.
    ```
  - Outcome:
    - Base bullets for the concepts section.

- Tool matrix
  - Prompt:
    ```text
    Create a conservative comparison matrix for Qdrant, Weaviate, Milvus, Pinecone, Chroma.
    Columns: hosting, hybrid, indexes, filtering, APIs, ops.
    ```
  - Outcome:
    - Draft matrix refined against public docs.

- Qdrant quickstart code
  - Prompt:
    ```text
    Provide Python client code to create a collection (size=384, cosine), upsert 3 points with payload,
    and run filtered query_points using qdrant-client.
    ```

- Hybrid fusion example
  - Prompt:
    ```text
    Show minimal RRF fusion using prefetch of sparse and dense named vectors with qdrant-client.
    ```

- Style guardrail
  - Prompt:
    ```text
    Verify that each ### subsection has a single details block, and block elements are indented
    correctly for Docusaurus rendering.
    ```

---

</details>
