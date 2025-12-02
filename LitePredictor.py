import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import List, Optional

from utils import dice_loss, iou_mse_loss

class LitePredictor:
    def __init__(self, model, device, image_size=256):
        self.model = model
        self.device = device
        self.image_size = image_size

        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _resize_box(self, box, original_size):
        """
        Rescale bbox coordinates from original size to model input size (256x256)
        """
        h, w = original_size
        scale_x = self.image_size / w
        scale_y = self.image_size / h

        x1, y1, x2, y2 = box
        return [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]

    def predict(self, image_path: str, bbox: List[int], threshold=0.5):
        self.model.eval()

        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # 2. Preprocess Image
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # 3. Rescale BBox to match the resized image
        resized_bbox = self._resize_box(bbox, (original_h, original_w))
        bbox_tensor = torch.tensor([resized_bbox], dtype=torch.float32).to(self.device)

        # 4. Predict
        with torch.no_grad():
            mask_pred, iou_pred = self.model(image_tensor, bbox_tensor)

            # --- FIX 1: Handle 5D output [1, 4, 1, 256, 256] ---
            if mask_pred.ndim == 5:
                # [Batch=0, Candidate=0, Channel=0, H, W]
                raw_mask = mask_pred[0, 0, 0]
            else:
                # [Batch=0, Candidate=0, H, W] (Standard fallback)
                raw_mask = mask_pred[0, 0]

            mask_prob = torch.sigmoid(raw_mask)

        # 5. Post-process (Resize mask back to original size)
        mask_prob_np = mask_prob.cpu().numpy()

        # verify shape is exactly 2D before resizing
        if mask_prob_np.ndim == 3:
             mask_prob_np = mask_prob_np.squeeze()

        mask_full_size = cv2.resize(mask_prob_np, (original_w, original_h),
                                   interpolation=cv2.INTER_LINEAR)

        final_mask = (mask_full_size > threshold).astype(np.uint8)

        return final_mask, iou_pred[0, 0].item()

    def visualize_prediction(self, image_path: str, bbox: List[int],
                            gt_mask_path: Optional[str] = None,
                            save_path: Optional[str] = None):
        """
        Visualize: Input+Box | Ground Truth | Prediction | Overlay
        """
        # Run prediction
        pred_mask, iou_score = self.predict(image_path, bbox)

        # Load Original Image for display
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Setup Plot
        cols = 4 if gt_mask_path else 3
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

        # 1. Input Image + BBox
        axes[0].imshow(image)
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                             fill=False, color='red', linewidth=3)
        axes[0].add_patch(rect)
        axes[0].set_title('Input Image + Prompt Box')
        axes[0].axis('off')

        # 2. Ground Truth
        plot_idx = 1
        if gt_mask_path:
            gt_mask = cv2.imread(gt_mask_path, 0)
            if gt_mask is None:
                gt_mask = np.zeros(image.shape[:2])

            if gt_mask.max() > 1:
                gt_mask = gt_mask / 255.0

            axes[plot_idx].imshow(gt_mask, cmap='gray')
            axes[plot_idx].set_title('Ground Truth Mask')
            axes[plot_idx].axis('off')
            plot_idx += 1

        

        # 3. Predicted Mask
        axes[plot_idx].imshow(pred_mask, cmap='gray')
        axes[plot_idx].set_title(f'Prediction')
        axes[plot_idx].axis('off')
        plot_idx += 1

        # 4. Overlay
        overlay = image.copy()
        overlay[pred_mask == 1] = [0, 255, 0]

        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        axes[plot_idx].imshow(blended)
        axes[plot_idx].set_title('Overlay Result')
        axes[plot_idx].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved result to {save_path}")
        plt.show()

    # def visualize_prediction(self, image_path: str, bbox: List[int],
    #                     gt_mask_path: Optional[str] = None,
    #                     save_path: Optional[str] = None):
    #     """
    #     Visualize: Input+Box | Ground Truth | Prediction | Overlay
    #     """
    #     # 1. Run prediction (Unpack 3 values now)
    #     pred_mask, pred_prob, pred_iou_value = self.predict(image_path, bbox)

    #     # Load Original Image
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Setup Plot
    #     cols = 4 if gt_mask_path else 3
    #     fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    #     # --- Plot 1: Input Image + BBox ---
    #     axes[0].imshow(image)
    #     rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
    #                             fill=False, color='red', linewidth=3)
    #     axes[0].add_patch(rect)
    #     axes[0].set_title(f'Input + Box\nPred IoU: {pred_iou_value:.3f}')
    #     axes[0].axis('off')

    #     # Variables for title
    #     dice_score = 0.0
    #     iou_mse_score = 0.0
        
    #     # --- Plot 2: Ground Truth (If available) ---
    #     plot_idx = 1
    #     if gt_mask_path:
    #         gt_mask = cv2.imread(gt_mask_path, 0)
    #         if gt_mask is None:
    #             gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
    #         # Binarize GT
    #         gt_mask = (gt_mask > 127).astype(np.uint8)

    #         # --- CALCULATE METRICS HERE ---
    #         # 1. Dice Score
    #         intersection = np.logical_and(pred_mask, gt_mask).sum()
    #         union_dice = pred_mask.sum() + gt_mask.sum()
    #         if union_dice > 0:
    #             dice_score = (2. * intersection) / union_dice
    #         else:
    #             dice_score = 1.0 if (pred_mask.sum() == 0 and gt_mask.sum() == 0) else 0.0

    #         # 2. Actual IoU (to calculate MSE)
    #         union_iou = np.logical_or(pred_mask, gt_mask).sum()
    #         actual_iou = intersection / (union_iou + 1e-6)

    #         # 3. IoU MSE (Predicted vs Actual)
    #         iou_mse_score = (pred_iou_value - actual_iou) ** 2

    #         axes[plot_idx].imshow(gt_mask, cmap='gray')
    #         axes[plot_idx].set_title('Ground Truth Mask')
    #         axes[plot_idx].axis('off')
    #         plot_idx += 1

    #     # --- Plot 3: Prediction ---
    #     axes[plot_idx].imshow(pred_mask, cmap='gray')
    #     if gt_mask_path:
    #         # Show calculated scores in title
    #         axes[plot_idx].set_title(f'Prediction\nDice: {dice_score:.3f} | IoU MSE: {iou_mse_score:.4f}')
    #     else:
    #         axes[plot_idx].set_title(f'Prediction')
    #     axes[plot_idx].axis('off')
    #     plot_idx += 1

    #     # --- Plot 4: Overlay (Optional, assuming cols=4) ---
    #     if plot_idx < cols:
    #         axes[plot_idx].imshow(image)
    #         axes[plot_idx].imshow(pred_mask, alpha=0.5, cmap='jet')
    #         axes[plot_idx].set_title('Overlay')
    #         axes[plot_idx].axis('off')

    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path)
    #         print(f"Saved visualization to {save_path}")
            
    #     plt.show()
        
    #     # Return metrics if you need to log them
    #     # return dice_score, iou_mse_score

def get_bbox_from_mask_file(mask_path):
    mask = cv2.imread(mask_path, 0)
    if mask is None: return [0, 0, 10, 10] # Dummy box if fail
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any(): return [0,0,10,10]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax, rmax] # x1, y1, x2, y2


