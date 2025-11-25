from utils import evaluate_batch, combined_loss
from tqdm import tqdm
import torch
from pathlib import Path



def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device) # (B, 3, H, W)
        masks_gt = batch['mask'].to(device) # (B, Mask_Num, 1, H, W)
        boxes = batch['bbox'].to(device)    # (B, Mask_Num, 4)
        
        # Forward pass
        masks_pred, iou_pred = model(images, boxes)
        
        # Compute loss
        loss = combined_loss(masks_pred, masks_gt)
        
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
):
    """Complete training pipeline"""
    Path(save_dir).mkdir(exist_ok=True)
    
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
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = evaluate_batch(model, val_loader, device)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Learning rate step
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'iou' : val_metrics["iou"]
            }, f"{save_dir}/best_model.pth")
            print(f"Saved best model with Dice: {best_dice:.4f}")
        
        if epoch%10 == 0 :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_metrics['dice'],
                'iou' : val_metrics["iou"]
            }, f"{save_dir}/best_model.pth")
    
    return history