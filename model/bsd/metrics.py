import torch

def results(pred, gt, threshold=0.5, eps=1e-8):
    """
    Compute accuracy, precision, recall, F1-score, and IoU between predicted and ground truth saliency maps.

    Args:
        pred (torch.Tensor): Predicted logits or probabilities, shape [B,1,H,W] or [1,H,W].
        gt (torch.Tensor): Ground truth binary mask, same shape as pred.
        threshold (float): Threshold to binarize predictions. Default=0.5.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        dict: {
            'accuracy': ...,
            'precision': ...,
            'recall': ...,
            'f1': ...,
            'iou': ...
        }
    """

    # Convert logits to probabilities if not already in [0,1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    # Binarize
    pred_bin = (pred > threshold).float()
    gt_bin = (gt > 0.5).float()

    # Flatten
    pred_flat = pred_bin.view(-1)
    gt_flat = gt_bin.view(-1)

    # Compute confusion matrix elements
    TP = (pred_flat * gt_flat).sum()
    FP = (pred_flat * (1 - gt_flat)).sum()
    FN = ((1 - pred_flat) * gt_flat).sum()
    TN = ((1 - pred_flat) * (1 - gt_flat)).sum()

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)

    # return {
    #     "accuracy": accuracy.item(),
    #     "precision": precision.item(),
    #     "recall": recall.item(),
    #     "f1": f1.item(),
    #     "iou": iou.item()
    # }
    return accuracy.item(), precision.item(), recall.item(), f1.item(), iou.item()
