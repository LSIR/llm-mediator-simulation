# OLMo-2 Reddit Fine-tuning

This subproject contains code for fine-tuning OLMo-2 32B on Reddit ChangeMyView data, specifically on pairs of comments that occurred just before derailment (violation of Rule 2: "Don't be rude or hostile to other users").

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up wandb for experiment tracking:
```bash
wandb login
```

## Training

The script will:
1. Load the Reddit comment pairs dataset
2. Initialize OLMo-2 32B with 4-bit quantization and LoRA
3. Fine-tune the model using the Hugging Face Trainer
4. Save checkpoints and the final model

To start training:
```bash
python train_olmo_reddit.py
```

## Model Details

- Base model: OLMo-2 32B
- Training method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit
- Batch size: 1 with gradient accumulation of 16
- Learning rate: 2e-4
- Training epochs: 3

## Dataset

The dataset consists of 1403 Reddit comment pairs from ChangeMyView:
- 840 training samples
- 291 validation samples
- 272 test samples

Each pair contains:
- The topic statement
- The penultimate comment before derailment
- The last comment before derailment 