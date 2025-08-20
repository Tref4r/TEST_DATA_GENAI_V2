"""
Modified inference script for the fine‑tuned QLoRA model.

This version aligns the inference prompt with the style used during
fine‑tuning ("User: …\nAssistant:") and removes unnecessary sampling
hyper‑parameters so that responses are deterministic.  It supports both
single‑question generation and a simple interactive loop.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(base_model_path: str, adapter_path: str):
    """Load the base model and merge the LoRA adapter for inference."""
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    max_memory = {0: "4500MB", "cpu": "24GB"}
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        offload_folder="offload",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(user_question: str) -> str:
    """Create a prompt matching the training format."""
    return f"User: {user_question}\nAssistant:"


def generate_response(model, tokenizer, user_question: str, max_new_tokens: int = 128):
    """Generate a deterministic answer for a single user question."""
    prompt = format_prompt(user_question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)
    return answer.strip()


def interactive_loop(model, tokenizer):
    """Run a simple REPL for the fine‑tuned assistant."""
    print("\nInteractive mode.  Type 'q' to quit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            break
        response = generate_response(model, tokenizer, user_input)
        print("Assistant:", response, "\n")


def main():
    base_model = "Qwen/Qwen1.5-1.8B-Chat"
    adapter_path = "outputs\checkpoint-3942"  # Adjust if your adapter is saved elsewhere
    print("Loading model…")
    model, tokenizer = load_model(base_model, adapter_path)
    # Single question demo
    test_questions = [
        "Năm xảy ra cuộc Cách mạng Pháp là năm nào?",  # Vietnamese question
        "How did Maximilien Robespierre die and when?",
    ]
    for q in test_questions:
        print(f"\nUser: {q}")
        print("Assistant:", generate_response(model, tokenizer, q))
    # Launch interactive loop
    interactive_loop(model, tokenizer)


if __name__ == "__main__":
    main()