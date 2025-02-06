import torch.nn as nn
import torch

def multi_task_loss(class_pred, class_target, bbox_pred, bbox_target):
    """
    Computes a combined loss for classification and bounding box regression.

    Args:
        class_pred (Tensor): Predicted class logits (shape [batch_size, num_classes]).
        class_target (Tensor): True class labels (shape [batch_size]).
        bbox_pred (Tensor): Predicted bounding boxes (shape [batch_size, 4]).
        bbox_target (Tensor): True bounding boxes (shape [batch_size, 4]).

    Returns:
        Tensor: Total loss combining classification and bounding box regression losses.
    """
    classification_loss = nn.CrossEntropyLoss()(class_pred, class_target)
    bbox_loss = nn.SmoothL1Loss()(bbox_pred, bbox_target)
    total_loss = classification_loss + bbox_loss
    return total_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
        # Loss for similar pairs
        loss_similar = labels * torch.square(distances)
        # Loss for dissimilar pairs
        loss_dissimilar = (1 - labels) * torch.square(torch.clamp(self.margin - distances, min=0))
        loss = 0.5 * torch.mean(loss_similar + loss_dissimilar)
        return loss
