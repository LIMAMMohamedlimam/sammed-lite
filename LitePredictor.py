
from typing import List, Optional
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import albumentations as A
import matplotlib.pyplot as plt

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
    
    def predict(self, image_path: str, bbox: List[int], threshold=0.5):
        """
        Predict segmentation mask
        
        Args:
            image_path: Path to input image
            bbox: Bounding box [x1, y1, x2, y2]
            threshold: Prediction threshold
        """
        self.model.eval()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        bbox_tensor = torch.tensor([bbox], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            mask_pred, iou_pred = self.model(image_tensor, bbox_tensor)
            mask_pred = torch.sigmoid(mask_pred[0, 0])
        
        # Post-process
        mask = (mask_pred > threshold).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        return mask, iou_pred.item()
    
    def visualize_prediction(self, image_path: str, bbox: List[int], 
                            save_path: Optional[str] = None):
        """Visualize prediction result"""
        # Predict
        mask, iou = self.predict(image_path, bbox)
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with bbox
        axes[0].imshow(image)
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                             fill=False, color='red', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title('Input + BBox')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Predicted Mask (IoU: {iou:.3f})')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        axes[2].imshow(blended)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
