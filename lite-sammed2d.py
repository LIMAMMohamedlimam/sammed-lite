
import torch
import torch.nn as nn
from adapter import Adapter
import torch.nn.functional as F

class SimpleSAMMed2D(nn.Module):
    """Simplified SAM-Med2D, SAM with adapter"""

    def __init__(self, sam_model, adapter_embed_dim: int = 768):
        super().__init__()
        self.sam = sam_model
        
        # Freeze image encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        
        # Add adapters to image encoder blocks
        self.adapters = nn.ModuleList([
            Adapter(adapter_embed_dim) 
            for _ in range(len(self.sam.image_encoder.blocks))
        ])
        
        # Fine-tune prompt encoder and mask decoder
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
    
    def forward(self, images, boxes):
        """
        Args:
            images: [B, 3, H, W]
            boxes: [B, 4] in xyxy format
        """
        batch_size = images.shape[0]
        
        # Image encoding with adapters
        image_embeddings = self._encode_with_adapters(images)
        
        # Prepare prompts (boxes)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        
        # Decode masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale masks
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        return masks, iou_predictions
    
    def _encode_with_adapters(self, x):
        """Image encoding with adapter injection"""
        x = self.sam.image_encoder.patch_embed(x)
        
        # Add positional encoding
        if self.sam.image_encoder.pos_embed is not None:
            x = x + self.sam.image_encoder.pos_embed
        
        # Pass through transformer blocks with adapters
        for i, block in enumerate(self.sam.image_encoder.blocks):
            x = block(x)
            x = self.adapters[i](x)  # Apply adapter
        
        x = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2))
        return x