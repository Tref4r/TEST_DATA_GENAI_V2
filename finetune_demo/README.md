# Fine-tuning Demo with QLoRA

This folder contains a practical demonstration of fine-tuning a large language model (LLM) using QLoRA (Quantized Low-Rank Adaptation) technique, optimized for hardware with limited VRAM (RTX 4050 6GB).

## Environment Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate finetune_demo
```

2. Verify CUDA setup:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
```

## Project Structure

- `environment.yml`: Conda environment configuration
- `train.py`: Main training script with QLoRA implementation
- `inference.py`: Script for inference using the fine-tuned model
- `outputs/`: Directory for saving model checkpoints and logs

## Implementation Details

### Hardware Requirements
- GPU: NVIDIA RTX 4050 (6GB VRAM)
- RAM: 32GB
- Storage: At least 30GB free space

### Optimization Techniques
1. **4-bit Quantization**
   - Uses NF4 format for weights
   - Reduces VRAM usage from ~14GB to ~4GB

2. **LoRA Parameters**
   - Rank (r): 16
   - Alpha: 32
   - Dropout: 0.05
   - Target: q_proj, k_proj, v_proj, o_proj modules

3. **Training Configuration**
   - Batch size: 4
   - Gradient accumulation steps: 4
   - Learning rate: 2e-4
   - Mixed precision (fp16)
   - 8-bit Adam optimizer

## Usage

1. Training:
```bash
python train.py
```

2. Inference:
```bash
python inference.py
```

## Memory Management

The implementation uses several techniques to fit within 6GB VRAM:
- 4-bit quantization (NF4)
- Gradient checkpointing
- Small batch size with gradient accumulation
- Flash Attention 2 (when available)

## Monitoring

- Training progress can be monitored through TensorBoard
- Memory usage is optimized for RTX 4050's 6GB VRAM limitation
- Basic error handling and logging implemented
