"""metrics.py


"""
from typing import Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score


def accuracy_torch(preds: torch.Tensor, truths: torch.Tensor) -> torch.Tensor: 

    return torch.sum(preds == truths)


def torch_metrics(preds: torch.Tensor, truths: torch.Tensor, epsilon: Optional[float] = 1e-7) -> Tuple[torch.Tensor]: 
    """torch_metrics

    Args:
        preds (torch.Tensor): prediction tensor
        truths (torch.Tensor): truth tensor
        epsilon (Optional[float], optional): smoothing term to avoid div by 0. Defaults to 1e-7.

    Returns:
        Tuple[torch.Tensor]: accuracy, f1, precision, recall of current batch
    """
    tp = (truths * preds).sum().to(torch.float32)
    tn = ((1 - truths) * (1 - preds)).sum().to(torch.float32)
    fp = ((1 - truths) * preds).sum().to(torch.float32)
    fn = (truths * (1 - preds)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision*recall) / (precision + recall + epsilon)

    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc, f1, precision, recall 