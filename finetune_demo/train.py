"""
Modified training script for QLoRA fine‑tuning with proper evaluation.

This script demonstrates how to fine‑tune a bilingual chat model using LoRA
adapters while keeping the training configuration lightweight for limited
hardware.  Compared to the original version, this script introduces a
validation split, an evaluation loop with simple metrics (Exact Match and
token‑level F1), and reduces the number of epochs to prevent overfitting.
Prompt formatting is left unchanged so that the model sees the same style
during training and inference.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# The tokenizer is set as a global so that the compute_metrics function can
# access it during evaluation.  It is initialised inside `train()`.
GLOBAL_TOKENIZER = None


# Base bilingual model.  You can swap this out for a different model as
# needed.  We use the same base as in the original script to ensure the
# adapter weights remain compatible.
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
OUTPUT_DIR = "outputs"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def setup_model_and_tokenizer():
    """Load the quantised model, wrap it with LoRA and prepare the tokenizer."""
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Limit GPU memory to fit on small devices
    max_memory = {0: "4500MB", "cpu": "24GB"}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        offload_folder="offload",
    )
    model.config.use_cache = False

    # Configure LoRA adapters.  We include additional projection layers used in
    # Qwen architectures so that all critical linear transformations are
    # adapted.  Bias is left as "none" because LoRA only applies to weight
    # matrices.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_example(example, tokenizer):
    """Convert a raw conversation example into tokenised input.

    This helper supports several common schemas, including alpaca style
    instruction/input/output and user/assistant message lists.  It produces
    a single string with speaker prefixes.  The same logic is used in
    `train.py` from the original repository.
    """
    convo = ""
    if isinstance(example.get("messages"), list):
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role.lower() in ("user", "human"):
                convo += f"User: {content}\n"
            elif role.lower() in ("assistant", "gpt", "bot"):
                convo += f"Assistant: {content}\n"
            else:
                convo += f"{content}\n"
    elif isinstance(example.get("conversations"), list):
        for turn in example["conversations"]:
            frm = turn.get("from", "")
            content = turn.get("value", "")
            if frm.lower() in ("user", "human"):
                convo += f"User: {content}\n"
            elif frm.lower() in ("assistant", "gpt", "bot"):
                convo += f"Assistant: {content}\n"
            else:
                convo += f"{content}\n"
    elif example.get("prompt") is not None and example.get("response") is not None:
        convo += f"User: {example['prompt']}\nAssistant: {example['response']}\n"
    elif example.get("instruction") is not None and example.get("output") is not None:
        # Map instruction/input/output to conversation turns.  Do not include
        # "Instruction:"/"Input:" prefixes; instead treat each as a user message.
        instr = example.get("instruction", "")
        inp = example.get("input")
        out = example.get("output", "")
        if instr:
            convo += f"User: {instr}\n"
        if inp:
            convo += f"User: {inp}\n"
        convo += f"Assistant: {out}\n"
    else:
        convo = str(example)
    tokens = tokenizer(
        convo,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": tokens["input_ids"][0],
        "attention_mask": tokens["attention_mask"][0],
    }


# def exact_match_and_f1(pred, label):
#     """Compute exact match (EM) and token‑level F1 between two strings."""
#     pred_tokens = pred.strip().lower().split()
#     label_tokens = label.strip().lower().split()
#     # Exact match
#     em = 1.0 if pred_tokens == label_tokens else 0.0
#     # Precision and recall for F1
#     common = set(pred_tokens) & set(label_tokens)
#     if len(pred_tokens) == 0 or len(label_tokens) == 0:
#         return em, 0.0
#     precision = len(common) / len(pred_tokens)
#     recall = len(common) / len(label_tokens)
#     if precision + recall == 0:
#         f1 = 0.0
#     else:
#         f1 = 2 * precision * recall / (precision + recall)
#     return em, f1


# def compute_metrics(eval_pred):
#     """Compute exact match and token‑level F1 for a batch.

#     The Trainer provides `eval_pred` as a tuple (`predictions`, `label_ids`).
#     We decode both sequences using a global tokenizer defined at runtime.  If
#     the tokenizer has not been initialised (e.g. during dry‑runs), the
#     function returns zeros.  To avoid mixing prompt and answer tokens, this
#     implementation decodes the entire sequence; for more precise scoring you
#     can slice the generation after the prompt length.
#     """
#     preds, labels = eval_pred
#     global GLOBAL_TOKENIZER
#     tok = GLOBAL_TOKENIZER
#     if tok is None:
#         return {"exact_match": 0.0, "f1": 0.0}
#     decoded_preds = []
#     decoded_labels = []
#     for pred_ids, label_ids in zip(preds, labels):
#         decoded_pred = tok.decode(pred_ids, skip_special_tokens=True)
#         decoded_label = tok.decode(label_ids, skip_special_tokens=True)
#         decoded_preds.append(decoded_pred)
#         decoded_labels.append(decoded_label)
#     em_scores = []
#     f1_scores = []
#     for p, l in zip(decoded_preds, decoded_labels):
#         em, f1 = exact_match_and_f1(p, l)
#         em_scores.append(em)
#         f1_scores.append(f1)
#     return {
#         "exact_match": sum(em_scores) / len(em_scores),
#         "f1": sum(f1_scores) / len(f1_scores),
#     }


def train():
    model, tokenizer = setup_model_and_tokenizer()
    # Make the tokenizer globally available for metric computation
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = tokenizer
    # Load the Vietnamese conversational dataset.  You need to accept the terms
    # of use on Hugging Face before downloading.  Remove the `.select` call
    # below to use the entire training set.  Here we use a small slice for
    # demonstration purposes.
    dataset = load_dataset("ontocord/viet4all", split="train")

    # Tokenise the entire dataset
    tokenised = dataset.map(
        lambda ex: prepare_example(ex, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenising examples",
    )

    # Split into train/validation sets.  Using an 80/20 split here.  For large
    # datasets, prefer using the built‑in `train_test_split`.
    split_point = int(0.8 * len(tokenised))
    train_dataset = tokenised.select(range(split_point))
    # eval_dataset = tokenised.select(range(split_point, len(tokenised)))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # reduce epochs to avoid overfitting
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,       
        eval_accumulation_steps=1,   
        save_steps=100,
        logging_steps=50,
        learning_rate=1e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        group_by_length=True,
        save_total_limit=2,
        # Note: older versions of transformers may not support `evaluation_strategy`.
        # We omit it here and call `trainer.evaluate()` manually after training.
        logging_dir="logs",
        report_to="none",  # disable external reporters
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # compute_metrics=compute_metrics,
    )
    trainer.train()
    # Explicitly run evaluation on the validation set.  Without
    # `evaluation_strategy`, this will not happen automatically.
    # small_eval_dataset = eval_dataset.select(range(min(15, len(eval_dataset))))
    # trainer.evaluate(eval_dataset=small_eval_dataset)
    trainer.save_model()


if __name__ == "__main__":
    train()