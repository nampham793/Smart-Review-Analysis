import torch.nn.functional as F
import torch
import torch.nn as nn

"""
Input:
    - Predictions and labels for a classification task.
    - Predictions and labels for a regression task.
    - Inputs and labels for a multi-class classification task.

Process:
    1. Define loss functions for binary classification, regression, and multi-class classification tasks.
    2. Compute and return the loss for each task based on the provided predictions and labels.

Output:
    - Loss values for each task (classification, regression, and multi-class classification) are printed.
"""

def loss_classifier(pred_classifier, labels_classifier):
    return nn.BCELoss()(pred_classifier, labels_classifier)

def loss_regressor(pred_regressor, labels_regressor):
    mask = labels_regressor != 0
    squared_error = (pred_regressor - labels_regressor)**2
    return (squared_error[mask].sum() / mask.sum()).mean()

def loss_softmax(inputs, labels, device):
    mask = labels != 0
    n, aspect, rate = inputs.shape
    loss = torch.zeros(labels.shape, device=device)
    for i in range(aspect):
        label_i = labels[:, i].clone()
        label_i[label_i != 0] -= 1
        label_i = label_i.type(torch.LongTensor).to(device)
        loss[:, i] = nn.CrossEntropyLoss(reduction='none')(inputs[:, i, :], label_i)
    return (loss[mask].sum() / mask.sum()).mean()

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    return loss

def bce_loss_weights(inputs, targets, weights):
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    weights = targets * (1 / weights.view(1, -1)) + (1 - targets) * (1 / (1 - weights.view(1, -1)))
    return (ce_loss * weights).mean()

def CB_loss(inputs, targets, samples_positive_per_cls, samples_negative_per_cls, no_of_classes=2, loss_type='sigmoid', beta=0.9999, gamma=2):
    samples_per_cls = torch.cat([samples_positive_per_cls.unsqueeze(-1), samples_negative_per_cls.unsqueeze(-1)], dim=-1)
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum(dim=-1).reshape(-1, 1) * no_of_classes
    weights = targets * weights[:, 0] + (1 - targets) * weights[:, 1]

    if loss_type == "focal":
        cb_loss = (sigmoid_focal_loss(inputs, targets) * weights).mean()
    elif loss_type == "sigmoid":
        cb_loss = (F.binary_cross_entropy(inputs, targets, reduction="none") * weights).mean()

    return cb_loss

if __name__ == "__main__":
    # Sample data for testing
    pred_classifier = torch.rand(10, 1)
    labels_classifier = torch.randint(0, 2, (10, 1)).float()

    pred_regressor = torch.rand(10, 1)
    labels_regressor = torch.rand(10, 1)

    inputs_softmax = torch.rand(10, 6, 5)
    labels_softmax = torch.randint(0, 6, (10, 6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Testing each function
    loss_cls = loss_classifier(pred_classifier, labels_classifier)
    loss_reg = loss_regressor(pred_regressor, labels_regressor)
    loss_soft = loss_softmax(inputs_softmax, labels_softmax, device)

    print("Loss Classifier:", loss_cls.item())
    print("Loss Regressor:", loss_reg.item())
    print("Loss Softmax:", loss_soft.item())