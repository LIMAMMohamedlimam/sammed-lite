from datetime import datetime
from utils import evaluate_batch, combined_loss
from tqdm import tqdm
import torch
from pathlib import Path
import os

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)       # (B, 3, H, W)
        masks_gt = batch['mask'].to(device)      # (B, Mask_Num, 1, H, W)
        boxes = batch['bbox'].to(device)         # (B, Mask_Num, 4)
        
        # Forward pass
        masks_pred, iou_pred = model(images, boxes)
        
        # Compute loss
        masks_pred_flat = masks_pred.flatten(0, 1)
        masks_gt_flat = masks_gt.flatten(0, 1)
        loss = combined_loss(masks_pred_flat, masks_gt_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_model(
    model, 
    train_loader, 
    val_loader,
    device, 
    num_epochs=50,
    learning_rate=1e-4,
    save_dir='checkpoints',
    checkpoint_frequency= 5
):
    """
    Complete training pipeline using custom Lite saving methods.
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': [], 'val_iou': []}
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {save_dir}")
    
    for epoch in range(1, num_epochs + 1):
        # 1. Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # 2. Validate
        val_metrics = evaluate_batch(model, val_loader, device)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        
        # 3. Learning rate step
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch}/{num_epochs} | Loss: {train_loss:.4f} | Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        
        # 4. Save BEST Model 
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            
            save_path = os.path.join(save_dir, "best_lite_model.pth")
            model.save_lite_weights(save_path)
            
            print(f"New Best Model (Dice: {best_dice:.4f}) saved to {save_path}")
        
        # 5. Periodic Checkpoint 
        if epoch % checkpoint_frequency == 0 or epoch == num_epochs:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            ckpt_filename = f"checkpoint_{timestamp}_epoch_{epoch}.pth"
            ckpt_path = os.path.join(save_dir, ckpt_filename)
            
            model.save_checkpoint(
                path=ckpt_path,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                scheduler=scheduler
            )
            print(f"Checkpoint saved to {ckpt_path}")
    
    print("Training Complete.")
    return history