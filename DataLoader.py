import json
import os
import numpy as np
from skimage.measure import label, regionprops
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import random
import matplotlib.pyplot as plt


class DatasetLoader(Dataset):
    """image dataset"""
    def __init__(
        self,
        data_dir:str,
        image_size: int = 256,
        mode: int = 1, # 0 for test , 1 for train,
        mask_num: int = 5,
        dataset_name:str=None
    ):
        data_type = {
            0 : 'test',
            1 : 'train'
        }
        
        self.image_size = image_size
        self.mask_num = mask_num
        mode = data_type[mode]

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        if dataset_name != None :
            dataset_name +="_"
        else:
            dataset_name="" 

        # Define transforms
        if mode == 'train':
            dataset = json.load(open(os.path.join(data_dir, f'{dataset_name}image2label_{mode}.json'), "r"))
            ## TODO : revoir la stratégie de normalisation
            self.transform = A.Compose([
                A.ToGray(p=1),
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=self.pixel_mean,
                           std=self.pixel_std),
                ToTensorV2(),
            ])
            self.image_paths = list(dataset.keys())
            raw_labels = list(dataset.values())
        else:
            dataset = json.load(open(os.path.join(data_dir, f'{dataset_name}label2image_{mode}.json'), "r"))

            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=self.pixel_mean,
                           std=self.pixel_std),
                ToTensorV2(),
            ])
            self.image_paths = list(dataset.values())
            raw_labels = list(dataset.keys())

        

        self.label_paths = []
        for label in raw_labels:
            if isinstance(label, str):
                self.label_paths.append([label])
            else:
                self.label_paths.append(label)

        print(f"{mode}ing dataset loaded!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # --- FIX STARTS HERE ---
        # 1. Retrieve the labels for the current index
        current_labels = self.label_paths[idx]

        # 2. Safety check (though your __init__ likely handles this)
        # Note: We use [current_labels] to make a list, not list(current_labels) 
        # which would split a string path into individual characters.
        if isinstance(current_labels, str):
            current_labels = [current_labels]
        # --- FIX ENDS HERE ---
            
        mask_paths = random.choices(current_labels, k=self.mask_num)
        
        masks_list = []
        boxes_list = []
        
        image_tensor = None 

        for m in mask_paths:
            pre_mask = cv2.imread(m, 0)
            
            if pre_mask is None:
                # Create blank mask if file not found
                pre_mask = np.zeros((h, w), dtype=np.uint8)
            elif pre_mask.shape[:2] != (h, w):
                # Resize if dimensions don't match image
                pre_mask = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)

 
            if pre_mask.max() > 1:
                pre_mask = pre_mask / 255.0
            
            # Apply transforms
            transformed = self.transform(image=image, mask=pre_mask)
            
            image_tensor = transformed['image']
            mask_tensor = transformed['mask'].to(torch.int64)
            
            # Get box
            box = self._get_bbox_from_mask(transformed['mask'])

            masks_list.append(mask_tensor)
            boxes_list.append(box)

        masks_tensor = torch.stack(masks_list).float().unsqueeze(1) 
        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)

        return {
            'image': image_tensor,
            'mask': masks_tensor,  
            'bbox': boxes_tensor,  
            'original_size': torch.tensor([h, w]),
            'image_path' : img_path,
            'masks_paths' : mask_paths
        }

    def __show_imag_mask__(self, idx):
        # Imports needed for visualization inside the method

        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        current_labels = self.label_paths[idx]
        if isinstance(current_labels, str):
            current_labels = [current_labels]
            
        if len(current_labels) > 0:
            mask_paths = random.choices(current_labels, k=self.mask_num)
        else:
            mask_paths = []

        print(f"DEBUG : mask_paths_len: {len(mask_paths) }mask_paths: {mask_paths}")
        
        masks_list = []

        for m in mask_paths:
            pre_mask = cv2.imread(m, 0)
            
            if pre_mask is None:
                pre_mask = np.zeros((h, w), dtype=np.uint8)
            elif pre_mask.shape[:2] != (h, w):
                pre_mask = cv2.resize(pre_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if pre_mask.max() > 1:
                pre_mask = pre_mask / 255.0

            masks_list.append(pre_mask)    

        
        total_plots = 1 + len(masks_list)
        
        plt.figure(figsize=(4 * total_plots, 4)) 

        # The Original Image
        plt.subplot(1, total_plots, 1)
        plt.imshow(image)
        plt.title(f"Image {img_path}")
        plt.axis('off')

        #  2...N: The Masks
        for i, mask in enumerate(masks_list):
            plt.subplot(1, total_plots, i + 2)
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1) 
            plt.title(f"Mask {i+1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _get_bbox_from_mask(mask):
        """Extract bounding box from mask"""
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return [0, 0, mask.shape[1], mask.shape[0]]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return [cmin, rmin, cmax, rmax]  # x1, y1, x2, y2


if __name__ == "__main__" :
    train_dataset = DatasetLoader("data_demo")
    print(train_dataset.__getitem__(0))