import torch
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
        if dense_embeddings.shape[-2:] != (H_e, W_e):
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=(H_e, W_e),
                mode='bilinear',
                align_corners=False
            )

        image_pe = self.sam.prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != (H_e, W_e):
            image_pe = F.interpolate(
                image_pe,
                size=(H_e, W_e),
                mode='bilinear',
                align_corners=False
            )
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings_expanded,
            image_pe=image_pe, 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        print(f"DEBUG: Mask Logits | Min: {low_res_masks.min().detach().item():.4f} | Max: {low_res_masks.max().detach().item():.4f}")

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
    





# Saving and loading methods ( training on limited resources )


    def save_lite_weights(self, path):
        """
        Saves ONLY the trainable parameters (adapters, mask_decoder, prompt_encoder).
        """
        # Filter state_dict to only include keys that required gradients
        trainable_keys = {
            k: v for k, v in self.state_dict().items() 
            if k in [n for n, p in self.named_parameters() if p.requires_grad]
        }
        torch.save(trainable_keys, path)
        print(f"Lite weights saved to {path} ({len(trainable_keys)} keys)")

    def load_lite_weights(self, path, device='cpu'):
        """
        Loads the lite weights into the model. 
        """
        state_dict = torch.load(path, map_location=device)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys found: {unexpected_keys}")
        print(f"Lite weights loaded successfully.")

    def save_checkpoint(self, path, optimizer, epoch, loss, scheduler=None):
        """
        Saves a training checkpoint containing the Lite model weights + Optimizer state.
        """
        trainable_model_state = {
            k: v for k, v in self.state_dict().items() 
            if k in [n for n, p in self.named_parameters() if p.requires_grad]
        }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainable_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: Epoch {epoch} at {path}")

    def load_checkpoint(self, path, optimizer, scheduler=None, device='cpu'):
        """
        Loads a checkpoint to resume training.
        Returns: epoch, loss
        """
        checkpoint = torch.load(path, map_location=device)
        
        # 1. Load Model
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 2. Load Optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 3. Load Scheduler 
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Resumed training from Epoch {epoch}")
        return epoch, loss