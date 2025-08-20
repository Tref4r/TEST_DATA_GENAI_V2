"""
Evaluate a fine-tuned QLoRA model on Vietnamese conversational dataset.
Compute simple metrics (Exact Match + token-level F1) without using Trainer.
Save aggregated results to a text file.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
# Config
# ======================
OUTPUT_DIR = "outputs/checkpoint-3942"   # checkpoint đã fine-tuned
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"    # base model (tokenizer)
MAX_EVAL_SAMPLES = 50                    # số mẫu tối đa để đánh giá
MAX_NEW_TOKENS = 128                     # số token sinh thêm khi generate
RESULTS_FILE = "eval_results.txt"        # file lưu kết quả


# ======================
# Helper functions
# ======================
def prepare_example(example, tokenizer):
    """Convert a raw conversation example into tokenised input."""
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
    elif example.get("prompt") and example.get("response"):
        convo += f"User: {example['prompt']}\nAssistant: {example['response']}\n"
    elif example.get("instruction") and example.get("output"):
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


def exact_match_and_f1(pred, label):
    """Compute exact match (EM) and token-level F1 between two strings."""
    pred_tokens = pred.strip().lower().split()
    label_tokens = label.strip().lower().split()
    em = 1.0 if pred_tokens == label_tokens else 0.0
    common = set(pred_tokens) & set(label_tokens)
    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return em, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(label_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return em, f1


# ======================
# Main evaluate script
# ======================
if __name__ == "__main__":
    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset("ontocord/viet4all", split="train")
    tokenised = dataset.map(
        lambda ex: prepare_example(ex, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenising examples",
    )

    # Split train/val
    split_point = int(0.8 * len(tokenised))
    eval_dataset = tokenised.select(range(split_point, len(tokenised)))

    # Giới hạn số lượng mẫu
    small_eval_dataset = eval_dataset.select(range(min(MAX_EVAL_SAMPLES, len(eval_dataset))))

    em_scores, f1_scores = [], []

    # Loop over eval samples
    for example in small_eval_dataset:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # decode prediction và label
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        em, f1 = exact_match_and_f1(pred, label)
        em_scores.append(em)
        f1_scores.append(f1)

    # Tính trung bình
    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # In ra console
    print("==================================================")
    print("Final Evaluation results on", len(small_eval_dataset), "samples")
    print("Exact Match:", avg_em)
    print("F1:", avg_f1)

    # Ghi ra file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Evaluation results\n")
        f.write(f"Samples: {len(small_eval_dataset)}\n")
        f.write(f"Exact Match: {avg_em:.4f}\n")
        f.write(f"F1: {avg_f1:.4f}\n")

    print(f"Saved results to {RESULTS_FILE}")
