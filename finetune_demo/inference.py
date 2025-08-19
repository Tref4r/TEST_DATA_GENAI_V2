import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model_path, adapter_path):
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
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        offload_folder="offload"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()  # Set to evaluation mode
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_prompt(instruction, input_text=None, chat_mode=False):
    if chat_mode:
        # Chế độ chat tự nhiên
        return f"### Instruction: Act as a helpful assistant and respond naturally to: {instruction}\n### Response:"
    else:
        # Format Alpaca chuẩn cho training
        if input_text:
            return f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: Let me provide a clear and concise answer."
        return f"### Instruction: {instruction}\n### Response: Let me provide a clear and concise answer."

def generate_response(model, tokenizer, instruction, input_text=None, max_length=512, chat_mode=False):
    try:
        # Format the prompt dựa vào mode
        prompt = format_prompt(instruction, input_text, chat_mode)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_input_length = len(inputs.input_ids[0])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,      # Giới hạn độ dài output ngắn hơn
                min_new_tokens=20,       # Đảm bảo câu trả lời đủ dài
                num_return_sequences=1,
                temperature=0.2,         # Giảm temperature để tập trung hơn
                do_sample=True,
                top_p=0.85,             # Giảm top_p để tăng tính chính xác
                top_k=40,               # Giảm top_k để tăng tính tập trung
                repetition_penalty=1.5,  # Tăng penalty cho việc lặp lại
                no_repeat_ngram_size=3,  
                length_penalty=0.8,      # Khuyến khích câu trả lời ngắn gọn
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0][max_input_length:], skip_special_tokens=True)
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Sorry, an error occurred during response generation."
    


def main():
    # Load model and adapter
    # Base model should mirror the one used during training.  For our chat
    # fine‑tuning we use the bilingual Qwen 1.5–1.8 B model.
    base_model = "Qwen/Qwen1.5-1.8B-Chat"
    adapter_path = "outputs"  # Path to your trained adapter
    
    print("Loading model...")
    model, tokenizer = load_model(base_model, adapter_path)
    
    while True:
        print("\n" + "="*50)
        print("Interactive Testing Mode")
        print("1. Run test cases")
        print("2. Interactive mode (Alpaca format)")
        print("3. Chat mode")
        print("4. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Test prompts - simpler cases first
            test_cases = [
                {
                    "instruction": "Tell me exactly the year, the course of this ",
                    "input": "What is the year that French Revolution occured ?"
                },
                {
                    "instruction": "Tell me exactly the year, the course of this",
                    "input": "How Maximilien Robespierre died? And when?"
                }
            ]
            
            print("\nGenerating responses for test cases...")
            for test in test_cases:
                print("\n" + "="*50)
                print(f"Instruction: {test['instruction']}")
                if test['input']:
                    print(f"Input: {test['input']}")
                response = generate_response(model, tokenizer, test['instruction'], test['input'])
                print(f"Output: {response}")
                print("="*50)
                
        elif choice == "2":
            while True:
                print("\n" + "="*50)
                instruction = input("Enter instruction (or 'q' to go back to main menu): ").strip()
                if instruction.lower() == 'q':
                    break
                    
                input_text = input("Enter input text (optional, press Enter to skip): ").strip()
                input_text = input_text if input_text else None
                
                print("\nGenerating response...")
                response = generate_response(model, tokenizer, instruction, input_text)
                print("\nOutput:", response)
                print("="*50)
                
        elif choice == "3":
            print("\n" + "="*50)
            print("Chat Mode - Talk naturally with the AI")
            print("Enter 'q' to return to main menu")
            print("="*50)
            
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'q':
                    break
                    
                print("\nAI:", end=" ")
                response = generate_response(model, tokenizer, user_input, chat_mode=True)
                print(response)
                
        elif choice == "4":
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
