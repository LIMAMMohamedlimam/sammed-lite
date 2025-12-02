## Utils file to ensure the data follows the SAM-MED 2D dataset conventions
import kagglehub
import os
import json
import random
import numpy as np
import cv2
from tqdm import tqdm


def down_finding_lungs_ct_data () :
    # Download latest version
    dataset_path = kagglehub.dataset_download("kmader/finding-lungs-in-ct-data")

    print("Path to dataset files:", dataset_path)

    # 1. Define your paths based on the screenshot
    base_dir = dataset_path
    images_dir = os.path.join(base_dir, "2d_images")
    masks_dir = os.path.join(base_dir, "2d_masks")

    # 2. Get list of filenames (filtering for .tif to be safe)
    # We assume filenames are identical in both folders as shown in your 'ls' command
    all_filenames = [f for f in os.listdir(images_dir) if f.endswith('.tif')]

    # 3. Shuffle the data for a random split
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(all_filenames)

    # 4. Calculate split index (80% for training)
    split_index = int(len(all_filenames) * 0.8)

    train_files = all_filenames[:split_index]
    test_files = all_filenames[split_index:]

    # 5. Create the dictionaries
    train_map = {}
    test_map = {}

    # Build Train Map: Image Path -> Mask Path
    for f in train_files:
        img_path = os.path.join(images_dir, f)
        mask_path = os.path.join(masks_dir, f)
        train_map[img_path] = mask_path

    # Build Test Map: Mask Path -> Image Path (As requested: label2image)
    for f in test_files:
        img_path = os.path.join(images_dir, f)
        mask_path = os.path.join(masks_dir, f)
        test_map[mask_path] = img_path

    # 6. Save to JSON files
    with open('lungs_ct_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)

    with open('lungs_ct_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print(f"Total files: {len(all_filenames)}")
    print(f"Training items (80%): {len(train_map)} saved to 'lungs_ct_image2label_train.json'")
    print(f"Testing items (20%): {len(test_map)} saved to 'lungs_ct_label2image_test.json'")











def down_lung_xrays_dataset() :

    # Download latest version
    path = kagglehub.dataset_download("nikhilpandey360/chest-xray-masks-and-labels")

    print("Path to dataset files:", path)

    # Define your paths
    base_dir = path + "/Lung Segmentation" 
    images_dir = os.path.join(base_dir, "CXR_png")
    masks_dir = os.path.join(base_dir, "masks")

    # Get list of filenames
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    all_mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

    print(f"Total images found: {len(all_image_files)}")
    print(f"Total masks found: {len(all_mask_files)}")

    # Match Images to Masks
    matched_samples = []

    print("Matching files...")

    # Create a set for faster lookup
    mask_files_set = set(all_mask_files)

    for img_file in all_image_files:
        image_id = os.path.splitext(img_file)[0]
        matching_masks = [m for m in all_mask_files if m.startswith(image_id)]
        if matching_masks:
            matched_samples.append((img_file, matching_masks))

    if len(matched_samples) == 0:
        print("WARNING: No matches found. Check your filenames to ensure image and mask names match.")
    else:
        print(f"Successfully matched {len(matched_samples)} pairs.")

    # Shuffle the data for a random split
    random.seed(42)
    random.shuffle(matched_samples)

    # Calculate split index (80% for training)
    split_index = int(len(matched_samples) * 0.8)

    train_samples = matched_samples[:split_index]
    test_samples = matched_samples[split_index:]

    # Create the dictionaries
    train_map = {}
    test_map = {}

    # Build Train Map: Image Path -> List of Mask Paths
    for img_file, mask_list in train_samples:
        img_path = os.path.join(images_dir, img_file)
        
        # Convert all mask filenames to full paths
        mask_paths = [os.path.join(masks_dir, m) for m in mask_list]
        
        train_map[img_path] = mask_paths

    # Build Test Map: Mask Path -> Image Path (label2image)
    for img_file, mask_list in test_samples:
        img_path = os.path.join(images_dir, img_file)
        
        # Flatten: Each mask points to its image
        for mask_file in mask_list:
            mask_path = os.path.join(masks_dir, mask_file)
            test_map[mask_path] = img_path

    # Save to JSON files
    with open('lung_xray_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)

    with open('lung_xray_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print("-" * 30)
    print(f"Total valid images found: {len(matched_samples)}")
    print(f"Training images: {len(train_map)} saved to 'lung_xray_image2label_train.json'")
    print(f"Testing masks: {len(test_map)} saved to 'lung_xray_label2image_test.json'")




def prepare_pancreas_npy_dataset(output_dir="Pancreas_Slices_Processed" , dataset_path=None):
    # --- 1. Locate Dataset ---
    # Based on your log, the path ends in 'versions/1'
    if dataset_path is None:
      try:
          dataset_path = kagglehub.dataset_download("tahsin/pancreasct-dataset")
          print(f"Dataset downloaded to: {dataset_path}")
      except:
          print("Could not download automatically. Please set 'dataset_path' manually.")
          return
    else:
      print(f"Using provided dataset_path: {dataset_path}")
   

    # Adjust paths for the structure seen in your logs: root/images/ and root/labels/
    images_source_dir = os.path.join(dataset_path, "images")
    labels_source_dir = os.path.join(dataset_path, "labels")

    # Output directories
    processed_imgs_dir = os.path.join(output_dir, "images")
    processed_masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(processed_imgs_dir, exist_ok=True)
    os.makedirs(processed_masks_dir, exist_ok=True)

    # --- 2. Helper: CT Windowing ---
    def apply_windowing(image, level=40, width=400):
        """
        Applies Soft Tissue Window (L:40, W:400) to raw CT data.
        """
        # Check if image is already normalized (approx 0-1) or raw (approx -1000 to 2000)
        if np.max(image) <= 1.0:
            return (image * 255).astype(np.uint8)
            
        lower = level - (width / 2)
        upper = level + (width / 2)
        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)
        return (image * 255).astype(np.uint8)

    # --- 3. Match Files ---
    print("Scanning for .npy volumes...")
    
    # Get all .npy files in images folder
    all_files = [f for f in os.listdir(images_source_dir) if f.endswith('.npy')]
    # Filter to ensure matching label exists
    matched_ids = [f for f in all_files if os.path.exists(os.path.join(labels_source_dir, f))]
    
    print(f"Found {len(matched_ids)} matched subjects (e.g., {matched_ids[0] if matched_ids else 'None'}).")

    # --- 4. Process Volumes ---
    sample_registry = [] 

    print("Slicing 3D .npy volumes into 2D images...")
    
    for filename in tqdm(matched_ids):
        # Load 3D arrays
        # Shape is usually (Height, Width, Depth) -> (512, 512, Slices)
        img_vol = np.load(os.path.join(images_source_dir, filename))
        lbl_vol = np.load(os.path.join(labels_source_dir, filename))
        
        # Validate shapes match
        if img_vol.shape != lbl_vol.shape:
            print(f"Skipping {filename}: Shape mismatch {img_vol.shape} vs {lbl_vol.shape}")
            continue

        # Determine Slice Axis (usually the smallest dimension or the one that isn't 512)
        # We assume the axis with the smallest size is the Depth (Z-axis)
        slice_axis = np.argmin(img_vol.shape)
        num_slices = img_vol.shape[slice_axis]
        
        # Move slice axis to the front for easier iteration: (Slices, H, W)
        img_vol = np.moveaxis(img_vol, slice_axis, 0)
        lbl_vol = np.moveaxis(lbl_vol, slice_axis, 0)

        subject_id = filename.replace('.npy', '')

        for i in range(num_slices):
            slice_img = img_vol[i]
            slice_mask = lbl_vol[i]

            # Filter: Keep slices with pancreas + small random subset of empty slices
            has_pancreas = np.sum(slice_mask) > 0
            
            if has_pancreas or random.random() < 0.1:
                # Resize to 256x256 for SAM-Med-Lite (Optional, but recommended for speed)
                slice_img = cv2.resize(slice_img, (256, 256))
                slice_mask = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                # Normalize/Window
                slice_img_uint8 = apply_windowing(slice_img)
                slice_mask_uint8 = (slice_mask > 0).astype(np.uint8) * 255

                # Save
                fname = f"{subject_id}_{i:04d}.png"
                cv2.imwrite(os.path.join(processed_imgs_dir, fname), slice_img_uint8)
                cv2.imwrite(os.path.join(processed_masks_dir, fname), slice_mask_uint8)

                sample_registry.append({"subject": subject_id, "filename": fname})

    # --- 5. Split & Save JSONs ---
    random.seed(42)
    unique_subjects = list(set([x['subject'] for x in sample_registry]))
    random.shuffle(unique_subjects)
    
    split_idx = int(len(unique_subjects) * 0.8)
    train_subs = set(unique_subjects[:split_idx])
    
    train_map, test_map = {}, {}

    for item in sample_registry:
        path_img = os.path.join(processed_imgs_dir, item["filename"])
        path_mask = os.path.join(processed_masks_dir, item["filename"])
        
        if item["subject"] in train_subs:
            train_map[path_img] = [path_mask]
        else:
            test_map[path_mask] = path_img

    with open('pancreas_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)
    with open('pancreas_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print(f"Done! Created {len(sample_registry)} slices from {len(unique_subjects)} patients.")
