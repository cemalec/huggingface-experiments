# Hugging Face Fine-Tuning Experiments

This repository contains scripts for fine-tuning transformer models on custom datasets and uploading the trained models to Hugging Face Hub.

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Hugging Face credentials for uploading models:
```bash
huggingface-cli login
# Or set the HF_TOKEN environment variable:
export HF_TOKEN=your_huggingface_token_here
```

## Quick Start

### Fine-Tune a Model

Use the `fine_tuning.py` script to fine-tune a transformer model on a dataset.

**Basic usage (default: BERT on MRPC dataset):**
```bash
python3 fine_tuning.py
```

**With custom model and dataset:**
```bash
python3 fine_tuning.py \
  --checkpoint bert-base-uncased \
  --dataset glue \
  --dataset-config mrpc
```

**Available dataset options:**
- `glue:mrpc` - Microsoft Research Paraphrase Corpus (default)
- `glue:stsb` - Semantic Textual Similarity Benchmark
- `glue:qqp` - Quora Question Pairs
- Other GLUE datasets or any Hugging Face dataset

**Arguments:**
- `--checkpoint`: Model checkpoint to fine-tune (default: `bert-base-uncased`)
- `--dataset`: Dataset name from Hugging Face (default: `glue`)
- `--dataset-config`: Dataset configuration/subset (default: `mrpc`)

**Output:**
The trained model checkpoints are saved in the `test-trainer/` directory:
```
test-trainer/
├── checkpoint-500/
├── checkpoint-690/
└── ...
```

Each checkpoint directory contains:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary
- Other training artifacts

---

### Upload a Model to Hugging Face Hub

Use the `upload_to_hub.py` script to upload your fine-tuned checkpoint to Hugging Face Hub.

**Interactive mode (select checkpoint):**
```bash
python3 upload_to_hub.py --repo-name username/my-model-name
```
This will list available checkpoints and let you choose which one to upload.

**Direct checkpoint path:**
```bash
python3 upload_to_hub.py \
  --checkpoint ./test-trainer/checkpoint-690 \
  --repo-name username/my-model-name
```

**With options:**
```bash
# Make the repository private
python3 upload_to_hub.py \
  --checkpoint ./test-trainer/checkpoint-690 \
  --repo-name username/my-model-name \
  --private

# Specify a different checkpoint directory
python3 upload_to_hub.py \
  --repo-name username/my-model-name \
  --checkpoint-dir ./model_checkpoints/test-trainer
```

**Upload arguments:**
- `--checkpoint`: Path to the checkpoint directory (if not provided, will prompt to select)
- `--repo-name`: Repository name on Hub in format `username/model-name` (required)
- `--private`: Make the repository private (optional)
- `--token`: Hugging Face API token (optional, uses `HF_TOKEN` env var or stored credentials by default)
- `--checkpoint-dir`: Directory containing checkpoints (default: `test-trainer`)

**Note:** The checkpoint path should be a directory, not a file. The script automatically finds the model weights and tokenizer files within the directory.

---

## Complete Workflow Example

1. **Fine-tune a model on MRPC:**
```bash
source venv/bin/activate
python3 fine_tuning.py --checkpoint bert-base-uncased --dataset glue --dataset-config mrpc
```

2. **Wait for training to complete** (checkpoints will be saved to `test-trainer/`)

3. **Upload the best checkpoint to Hub:**
```bash
python3 upload_to_hub.py --checkpoint ./test-trainer/checkpoint-690 --repo-name cemalec/my-paraphrase-model
```

4. **Access your model on Hub:**
Your model will be available at `https://huggingface.co/cemalec/my-paraphrase-model`

You can then load it with:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("cemalec/my-paraphrase-model")
tokenizer = AutoTokenizer.from_pretrained("cemalec/my-paraphrase-model")
```

---

## Files

- `fine_tuning.py` - Script to fine-tune transformer models
- `upload_to_hub.py` - Script to upload checkpoints to Hugging Face Hub
- `load_model.py` - Utilities for loading pre-trained models
- `utils.py` - Helper functions (e.g., device detection)
- `requirements.txt` - Python dependencies

---

## Recommended Datasets for Name Comparison

If you're working with name comparison tasks, consider these datasets:

1. **MRPC** (default) - Good for semantic similarity and paraphrase detection
2. **STS Benchmark** - Provides similarity scores (0-5 range)
3. **QQP** - Good for duplicate/similar name detection
4. **Custom Dataset** - Create your own with name pairs and labels

---

## Troubleshooting

### Model upload fails with authentication error
Make sure you've set up credentials:
```bash
huggingface-cli login
# Or set the environment variable:
export HF_TOKEN=your_token
```

### Checkpoint directory not found
Ensure the checkpoint path points to an actual directory containing model files like `model.safetensors` or `pytorch_model.bin`.

### Out of memory errors during fine-tuning
Reduce `per_device_train_batch_size` in `fine_tuning.py` (e.g., change from 16 to 8).

---

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Model Hub](https://huggingface.co/models)
