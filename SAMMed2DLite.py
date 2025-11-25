import torch.nn as nn
from segment_anything.modeling import Sam
import torch.nn.functional as F

class SAMMed2DLite(nn.Module):
    def __init__(
        self,
        sam_model : Sam,
        embed_dim=768,
    ):
        """Lite version of SAM-Med2D with adapter layers for medical image segmentation."""
        super().__init__()
        self.sam = sam_model

        # Freezing image encoder params except the adapter layer
        for name, params in self.sam.image_encoder.named_parameters():
            if "adapter" in name.lower():
                params.requires_grad = True
            else: 
                params.requires_grad = False
            
        # Fine-tune prompt encoder and mask decoder
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

    def forward(self, images, boxes):
        """
        Args:
            images: [B, 3, H, W]
            boxes:  [B, K, 4] where K is the number of masks per image
        """
        # 1. Encode Images
        image_embeddings = self.sam.image_encoder(images)

        # 2. Handle Dimensions
        B, C, H_e, W_e = image_embeddings.shape
        K = boxes.shape[1]
        
        # Flatten boxes from [B, K, 4] -> [B*K, 4]
        boxes_flat = boxes.view(-1, 4)

        # Repeat Image Embeddings to match the number of boxes
        image_embeddings_expanded = image_embeddings.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H_e, W_e)

        # 3. Encode Prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes_flat, 
            masks=None,
        )
        
        # 4. Decode Masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings_expanded,
            image_pe=self.sam.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 5. Upscale Masks
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # 6. Reshape back to [B, K, 1, H, W] if you want to compute loss per image
        masks = masks.view(B, K, 1, images.shape[2], images.shape[3])
        iou_predictions = iou_predictions.view(B, K, 1)
        
        return masks, iou_predictions