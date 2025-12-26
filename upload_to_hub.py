import logging
import argparse
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir: str = "test-trainer") -> list:
    """
    List all available checkpoints in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of checkpoint paths
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = sorted([
        d for d in checkpoint_path.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint-")
    ], key=lambda x: int(x.name.split("-")[1]))
    
    return checkpoints


def select_checkpoint(checkpoint_dir: str = "test-trainer") -> str:
    """
    Interactively select a checkpoint from available options.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to selected checkpoint
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        logger.error("No checkpoints found")
        return None
    
    logger.info("Available checkpoints:")
    for i, ckpt in enumerate(checkpoints, 1):
        logger.info(f"  {i}. {ckpt.name}")
    
    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(checkpoints):
                selected = checkpoints[index]
                logger.info(f"Selected checkpoint: {selected}")
                return str(selected)
            else:
                logger.warning(f"Please enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            logger.warning("Please enter a valid number")


def upload_checkpoint(
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
    token: str = None
) -> bool:
    """
    Upload a fine-tuned checkpoint to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        repo_name: Name of the repository on Hub (e.g., "username/model-name")
        private: Whether the repository should be private
        token: Hugging Face API token (if not provided, will use stored credentials)
        
    Returns:
        True if upload successful, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint path not found: {checkpoint_path}")
        return False
    
    try:
        # Login if token provided
        if token:
            logger.info("Logging in with provided token")
            login(token=token)
        
        # Load model and tokenizer
        logger.info(f"Loading model from {checkpoint_path}")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Push to Hub
        logger.info(f"Pushing model to {repo_name}")
        model.push_to_hub(
            repo_name,
            private=private,
            commit_message=f"Upload fine-tuned model from {checkpoint_path.name}"
        )
        
        logger.info(f"Pushing tokenizer to {repo_name}")
        tokenizer.push_to_hub(
            repo_name,
            private=private,
            commit_message=f"Upload tokenizer from {checkpoint_path.name}"
        )
        
        logger.info(f"Successfully uploaded to https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading checkpoint: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload a fine-tuned checkpoint to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (if not provided, will prompt to select)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name of the repository on Hub (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (if not provided, will use stored credentials)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="test-trainer",
        help="Directory containing checkpoints (default: test-trainer)"
    )
    
    args = parser.parse_args()
    
    # Select checkpoint if not provided
    if args.checkpoint is None:
        args.checkpoint = select_checkpoint(args.checkpoint_dir)
        if args.checkpoint is None:
            logger.error("No checkpoint selected")
            return
    
    # Validate checkpoint path
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint path does not exist: {args.checkpoint}")
        return
    
    # Upload to Hub
    success = upload_checkpoint(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token
    )
    
    if success:
        logger.info("Upload completed successfully!")
    else:
        logger.error("Upload failed")


if __name__ == "__main__":
    main()
