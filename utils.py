import torch
import tqdm
import torch.nn.functional as F



## Evaluation functions


def compute_dice_coefficient(pred, target, threshold=0.5):
    """Compute Dice coefficient"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice.item()

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()

def evaluate_batch(model, dataloader, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks_gt = batch['mask'].to(device)
            boxes = batch['bbox'].to(device)
            
            # Forward pass
            masks_pred, _ = model(images, boxes)
            
            # Compute metrics
            for i in range(images.shape[0]):
                dice = compute_dice_coefficient(masks_pred[i], masks_gt[i])
                iou = compute_iou(masks_pred[i], masks_gt[i])
                total_dice += dice
                total_iou += iou
                num_samples += 1
    
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    
    return {'dice': avg_dice, 'iou': avg_iou}



## loss functions

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for binary segmentation.
    Args:
        pred: logits (B, 1, H, W)
        target: binary mask (B, 1, H, W)
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Args:
        pred: logits (B, 1, H, W)
        target: binary mask (B, 1, H, W)
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def iou_mse_loss(pred_mask, gt_mask, pred_iou):
    """
    MSE loss between predicted IoU score and true IoU.
    Args:
        pred_mask: logits (B, 1, H, W)
        gt_mask: ground-truth mask (B, 1, H, W)
        pred_iou: predicted IoU head output (B,)
    """
    pred_mask = torch.sigmoid(pred_mask)

    intersection = (pred_mask * gt_mask).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3)) - intersection

    true_iou = (intersection + 1e-6) / (union + 1e-6)
    return F.mse_loss(pred_iou.squeeze(), true_iou)


def combined_loss(pred, target, focal_w=20.0, dice_w=1.0):
    """
    Hybrid loss combining Dice and Focal loss.
    Matches the paper ratio: 20 (Focal) : 1 (Dice)
    """
    d = dice_loss(pred, target)
    f = focal_loss(pred, target)
    return dice_w * d + focal_w * f


def total_loss_fn(pred_mask, gt_mask, pred_iou):
    """
    Full loss = Mask Loss (Focal+Dice) + IoU MSE loss.
    """
    mask_loss = combined_loss(pred_mask, gt_mask)
    iou_loss = iou_mse_loss(pred_mask, gt_mask, pred_iou)
    return mask_loss + iou_loss
