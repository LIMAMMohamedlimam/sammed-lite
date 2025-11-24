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

        #Freezing image encoder params except the adapter layer
        for name , params in self.sam.image_encoder.named_parameters() :
            if "adapter" in name.lower():
                params.requires_grad = True
            else : 
                params.requires_grad = False
            
        # Fine-tune prompt encoder and mask decoder
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

    def forward (self,images,boxes) :
        """
        Args : 
            images : [B, 3, H, W]
            boxes : [B, 4] in xyxy format
        """

        batch_size = images.shape[0]

        #image encoding 
        image_embeddings = self.sam.image_encoder.forward(images)

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


