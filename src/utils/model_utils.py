"""
Model utilities for sentiment analysis with DistilBERT.
Provides helper functions for model loading, saving, and inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Optional


def get_device() -> torch.device:
    """
    Get the appropriate device (GPU or CPU) for model execution.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def load_model(model_name: str, num_labels: int = 2, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a pretrained DistilBERT model for sequence classification.
    
    Args:
        model_name: Name or path of the pretrained model (e.g., 'distilbert-base-uncased').
        num_labels: Number of classification labels (default: 2 for binary sentiment).
        device: Device to load the model to. If None, uses GPU if available.
    
    Returns:
        Tuple of (model, device).
    """
    if device is None:
        device = get_device()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    return model, device


def load_tokenizer(model_name: str) -> object:
    """
    Load a tokenizer for the specified model.
    
    Args:
        model_name: Name or path of the pretrained model.
    
    Returns:
        Loaded tokenizer object.
    """
    return AutoTokenizer.from_pretrained(model_name)


def save_model(model: torch.nn.Module, tokenizer: object, save_path: str) -> None:
    """
    Save a model and its tokenizer to disk.
    
    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        save_path: Directory path where the model and tokenizer will be saved.
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def load_saved_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a previously saved model from disk.
    
    Args:
        model_path: Path to the saved model directory.
        device: Device to load the model to. If None, uses GPU if available.
    
    Returns:
        Tuple of (model, device).
    """
    if device is None:
        device = get_device()
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    return model, device


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Get information about a model (number of parameters, trainable parameters, etc.).
    
    Args:
        model: The model to get info about.
    
    Returns:
        Dictionary containing model information.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def freeze_backbone(model: torch.nn.Module) -> None:
    """
    Freeze all parameters in the model backbone (DistilBERT layers).
    Useful for transfer learning when you only want to train the classification head.
    
    Args:
        model: The model whose backbone should be frozen.
    """
    for param in model.distilbert.parameters():
        param.requires_grad = False
    print("Model backbone frozen for transfer learning")


def unfreeze_backbone(model: torch.nn.Module) -> None:
    """
    Unfreeze all parameters in the model backbone.
    
    Args:
        model: The model whose backbone should be unfrozen.
    """
    for param in model.distilbert.parameters():
        param.requires_grad = True
    print("Model backbone unfrozen")
