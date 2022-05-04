import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def saliency(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> np.ndarray: 
    """saliency

    Args:
        model (nn.Module): model for inference
        X (torch.Tensor): input tensor
        y (torch.Tensor): output tensor

    Returns:
        saliency map (np.ndarray)
    """
    model.eval()

    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)

    prediction = model(X_var)

    prediction.backward(y_var.double())

    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, i = torch.max(saliency,dim=1)
    saliency = saliency.squeeze() 

    #	print saliency.shape
    return saliency.data