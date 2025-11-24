import json 
import os 
import random
import numpy as np
from skimage.measure import label, regionprops
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch import Dataset
import albumentations as A

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)



class DatasetLoader (Dataset):
    """image dataset"""
    def __init__(
        self, 
        data_dir:str,
        image_size: int = 256,
        mode : int = 1 , # 0 for test , 1 for train
    ):
        data_type = {
            0 : 'test',
            1 : 'train'
        }

        self.image_size = image_size
        mode = data_type[mode]
        
        # Define transforms
        if mode ==  'train':
            dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))

            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            dataset = json.load(open(os.path.join(data_dir, f'label2image_{mode}.json'), "r"))

            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())

        print(f"{mode}ing dataset loaded!")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.label_paths 
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Generate bounding box from mask
        bbox = self._get_bbox_from_mask(mask.numpy())
        
        return {
            'image': image,
            'mask': torch.from_numpy(mask).float().unsqueeze(0) / 255.0,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'original_size': (image.shape[1], image.shape[2])
        }
    
    @staticmethod
    def _get_bbox_from_mask(mask):
        """Extract bounding box from mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, mask.shape[1], mask.shape[0]]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [cmin, rmin, cmax, rmax]  # x1, y1, x2, y2