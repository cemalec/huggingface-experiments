"""
Utility functions for Hugging Face models
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device():
    """
    Detect and return the appropriate device for processing.
    
    Returns:
        torch.device: The appropriate device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
    """
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) detected and available")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("CUDA GPU detected")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")