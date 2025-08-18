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
MODEL_NAME = "facebook/opt-350m"  # Smaller model that's easier to work with
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # For Mistral architecture
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_dataset(examples, tokenizer):
    # Format the text for training
    prompt = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output}"
        prompt.append(text)

    # Tokenize the text
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    return tokenized

def train():
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # Load and prepare dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")  # Use more training samples
    
    # Process the dataset
    processed_dataset = dataset.map(
        lambda examples: prepare_dataset(examples, tokenizer),
        batched=True,
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
        report_to=None         # Disable tensorboard to avoid dependency issues
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
