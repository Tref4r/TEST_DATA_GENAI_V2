import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# Constants
# Base model for fine‑tuning.  We choose a small bilingual model that fits a
# 6 GB GPU when quantized with QLoRA.  For Vietnamese/English chat we use
# the Qwen 1.5–1.8 B chat model.  You can swap this for a different base
# model if desired.
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
OUTPUT_DIR = "outputs"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def setup_model_and_tokenizer():
    # 4-bit quantization configuration
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Custom device map for memory management
    max_memory = {0: "4500MB", "cpu": "24GB"}
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        offload_folder="offload"
    )
    model.config.use_cache = False
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        # For Qwen architectures we include additional projection layers used in
        # the gating and feed-forward blocks.  This ensures LoRA adapts all
        # relevant linear transformations.
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

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_example(example, tokenizer):
    """
    Normalize a single data sample into tokenized input suitable for
    causal language model fine‑tuning.  This helper tries to support
    several common dataset schemas:

    * ``messages``: a list of dictionaries with ``role`` (``user``/``assistant``)
      and ``content``; we concatenate these turns with role prefixes.
    * ``conversations``: a list of dictionaries with ``from`` (``human`` or
      ``gpt``/``assistant``) and ``value``; similar concatenation is used.
    * ``prompt``/``response``: used in some instruct datasets; we turn into
      a two‑turn dialogue.
    * ``instruction``/``input``/``output``: Alpaca‑style format; we convert to
      a basic multi‑turn conversation.

    If none of these fields are present, the raw string representation
    of the example is used as input.
    """
    convo = ""
    # 1) messages-style structure
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
    # 2) conversations-style structure (sharegpt format)
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
    # 3) prompt/response pairs
    elif example.get("prompt") is not None and example.get("response") is not None:
        convo += f"User: {example['prompt']}\nAssistant: {example['response']}\n"
    # 4) Alpaca-style instruction/input/output
    elif example.get("instruction") is not None and example.get("output") is not None:
        instr = example.get("instruction", "")
        inp = example.get("input")
        out = example.get("output", "")
        if inp:
            convo += f"User: {instr}\nInput: {inp}\nAssistant: {out}\n"
        else:
            convo += f"User: {instr}\nAssistant: {out}\n"
    # Fallback
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

def train():
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Load the Viet4All dataset.  It is a Vietnamese translation and
    # subsampling of the OpenHermes‑2.5 corpus【318406737736123†L78-L93】.  You must
    # accept the terms of use on HuggingFace before downloading this
    # dataset.  We default to the ``train`` split; adjust or add other
    # splits as needed.
    dataset = load_dataset(
        "ontocord/viet4all",
        split="train",
    )

    # Process the dataset into tokenized examples.  We use ``select`` to
    # sample a small subset for demonstration; remove ``select`` to use the
    # entire training set.  The ``map`` call applies ``prepare_example``
    # to each entry individually.
    processed_dataset = dataset.select(range(1000)).map(
        lambda example: prepare_example(example, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )

    # Training arguments optimized for 6GB VRAM with improved training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50,  # Increased epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-4,  # Reduced learning rate for stability
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.5,  # Increased for better gradient flow
        warmup_ratio=0.1,   # Increased warmup
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        group_by_length=True,  # Group similar lengths for efficiency
        save_total_limit=3,    # Keep only the last 3 checkpoints
        logging_dir="logs",    # Add tensorboard logging
        report_to="tensorboard"         # Disable tensorboard to avoid dependency issues
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model()

if __name__ == "__main__":
    train()
