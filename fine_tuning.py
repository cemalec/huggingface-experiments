import logging
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
from utils import get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def main(checkpoint: str, dataset_name: str, dataset_config: str):
    """
    Main function to perform fine-tuning on a Hugging Face model.
    
    Args:
        checkpoint: The model checkpoint to use (default: bert-base-uncased)
        dataset_name: The dataset name (default: glue)
        dataset_config: The dataset configuration (default: mrpc)
    """
    # Get the appropriate device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load the dataset
    logger.info(f"Loading {dataset_name} {dataset_config} dataset")
    raw_datasets = load_dataset(dataset_name, dataset_config)
    
    # Initialize the tokenizer
    logger.info(f"Loading tokenizer from checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Tokenize the dataset
    logger.info("Tokenizing dataset")
    tokenized_datasets = raw_datasets.map(
        lambda example: tokenize_function(example, tokenizer),
        batched=True
    )
    
    # Create data collator
    logger.info("Creating data collator")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Configure training arguments
    logger.info("Configuring training arguments")
    training_args = TrainingArguments(
        "model_checkpoints/test-trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    # Load the model
    logger.info(f"Loading model from checkpoint: {checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    # Initialize trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    logger.info("Fine-tuning setup complete")
    
    # Start training and evaluation
    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")
    
    logger.info("Starting evaluation")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bert-base-uncased",
        help="The model checkpoint to use (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="glue",
        help="The dataset name (default: glue)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="mrpc",
        help="The dataset configuration (default: mrpc)"
    )
    
    args = parser.parse_args()
    
    main(
        checkpoint=args.checkpoint,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config
    )