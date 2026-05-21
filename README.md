# UGER-Prompter

Uncertainty-Gated Experience Retrieval for Chinese Polyphone Disambiguation

## Overview

UGER-Prompter is a three-stage framework for Chinese polyphone disambiguation:

1. **Stage 1: SFT (Supervised Fine-Tuning)** - Train base model with LoRA on position-first reading prediction
2. **Stage 2: Memory Building** - Build experience pool from scored predictions and generate hint templates
3. **Stage 3: RL (GRPO)** - Reinforcement learning with Group Relative Policy Optimization

## Directory Structure

```
UGER-Prompter/
├── configs/
│   └── config.json              # Global configuration
├── stage1_sft/
│   ├── build_sft_data.py        # Build SFT training data
│   └── train_sft.py             # SFT training with LoRA
├── stage2_memory/
│   ├── score_candidates.py      # Score predictions with SFT model
│   ├── build_experience_pool.py # Build UGER experience pool
│   └── build_hint_templates.py  # Generate hint templates
├── stage3_rl/
│   └── train_grpo.py            # GRPO training
├── evaluation/
│   ├── predict.py               # Batch prediction
│   └── evaluate.py              # Evaluation metrics
└── tools/                       # Utility tools
```

## Quick Start

### 1. Build SFT Data

```bash
python stage1_sft/build_sft_data.py \
    --input data/train.jsonl \
    --candidates data/candidates.jsonl \
    --output runs/sft_data.jsonl
```

### 2. Train SFT Model

```bash
python stage1_sft/train_sft.py \
    --data runs/sft_data.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output runs/sft_lora
```

### 3. Score Candidates

```bash
python stage2_memory/score_candidates.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter runs/sft_lora/final_model \
    --data runs/sft_data.jsonl \
    --output runs/scored.jsonl
```

### 4. Build Experience Pool

```bash
python stage2_memory/build_experience_pool.py \
    --scored runs/scored.jsonl \
    --output runs/experience_pool \
    --entropy-threshold 0.3
```

### 5. Build Hint Templates

```bash
python stage2_memory/build_hint_templates.py \
    --pool runs/experience_pool/experience_pool.jsonl \
    --kg data/knowledge_graph.jsonl \
    --output runs/hint_templates
```

### 6. Train GRPO

```bash
python stage3_rl/train_grpo.py \
    --data runs/sft_data.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --sft-adapter runs/sft_lora/final_model \
    --hints runs/hint_templates/hint_templates.jsonl \
    --output runs/rl_grpo
```

### 7. Evaluate

```bash
python evaluation/predict.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter runs/rl_grpo/final_model \
    --data data/test.jsonl \
    --output runs/predictions.jsonl

python evaluation/evaluate.py \
    --predictions runs/predictions.jsonl \
    --output runs/metrics.json
```

## Configuration

Edit `configs/config.json` to customize:

- Model settings (base model, max length)
- SFT hyperparameters (learning rate, LoRA config)
- Memory thresholds (entropy, support count)
- RL parameters (group size, reward weights)

## Key Features

- **Position-first prediction**: Each target is processed independently
- **Uncertainty-gated retrieval**: High-entropy predictions trigger memory retrieval
- **Lightweight hints**: Natural language hints for prompt injection
- **GRPO training**: Group-relative advantage estimation without critic network

## Citation

```bibtex
@article{xue2025uger,
  title={UGER-Prompter: Uncertainty-Gated Experience Retrieval for Chinese Polyphone Disambiguation},
  author={Xue, Jianxin},
  journal={Knowledge-Based Systems},
  year={2025}
}
```
