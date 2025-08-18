# Vector Database Tutorial Implementation

This repository contains a practical implementation of vector databases using Qdrant, optimized for systems with NVIDIA RTX 4050 GPU (6GB VRAM).

## Environment Setup

1. Create conda environment:
```bash
conda create -n vector_db python=3.11
conda activate vector_db
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/` - Core implementation files
  - Database setup and configuration
  - Vector operations and utilities
- `examples/` - Example implementations and use cases
- `docs/` - Documentation and guides

## GPU Configuration

This implementation is optimized for:
- NVIDIA RTX 4050 (6GB VRAM)
- 32GB System RAM

## Getting Started

1. Install and start Qdrant:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Run example implementations from the `examples/` directory
