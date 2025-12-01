## Utils file to ensure the data follows the SAM-MED 2D dataset conventions
import kagglehub
import os
import json
import random


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








def down_kits23_sample() :

    # Download latest version
    path = kagglehub.dataset_download("pawankumar1246/sample")

    print("Path to dataset files:", path)

    

    # Define your paths
    kits_dataset_path = ".." + path + "/AUGMENTED/DATASET_FINAL/"

    base_dir = kits_dataset_path
    images_dir = os.path.join(base_dir, "JPEGImages")
    masks_dir = os.path.join(base_dir, "Annotations")

    # Get list of filenames
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
    all_mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

    print(f"Total images: {len(all_image_files)}")
    print(f"Total masks: {len(all_mask_files)}")

    # Match Images to Masks
    matched_samples = []

    print("Matching files...")

    for img_file in all_image_files:
        if "orig" in img_file :
            # Extract the ID from the image filename (e.g., "case_001.jpg" -> "case_001")
            image_id = os.path.splitext(img_file)[0]

            # Find all masks that start with this ID followed by an underscore
            matching_masks = [m for m in all_mask_files if m.startswith(image_id + "_")]

            if len(matching_masks) > 0:
                # We append the image and the LIST of all matching masks
                matched_samples.append((img_file, matching_masks))
                if len(matching_masks) > 1:
                    print(f"Info: Image {img_file} has {len(matching_masks)} masks.")
            else:
                print(f"Warning: No mask found for image {img_file}. Skipping.")


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
    # Since one image can have multiple masks, we store the masks as a list.
    for img_file, mask_list in train_samples:
        img_path = os.path.join(images_dir, img_file)

        # Convert all mask filenames to full paths
        mask_paths = [os.path.join(masks_dir, m) for m in mask_list]

        train_map[img_path] = mask_paths

    # Build Test Map: Mask Path -> Image Path (label2image)
    for img_file, mask_list in test_samples:
        img_path = os.path.join(images_dir, img_file)

        for mask_file in mask_list:
            mask_path = os.path.join(masks_dir, mask_file)
            test_map[mask_path] = img_path

    # Save to JSON files
    with open('kits23_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)

    with open('kits23_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print("-" * 30)
    print(f"Total valid images found: {len(matched_samples)}")
    print(f"Training images: {len(train_map)} saved to 'kits23_image2label_train.json'")
    print(f"Testing masks: {len(test_map)} saved to 'kits23_label2image_test.json'")









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







    import os
import json
import random
import numpy as np
import nibabel as nib  # For loading NIfTI (3D CT files)
import cv2
import kagglehub
from tqdm import tqdm

def prepare_pancreas_ct_dataset(output_dir="Pancreas_Slices_Processed"):
    # Download Dataset (or set path if you have it locally) ---
    print("Downloading Pancreas-CT dataset...")
    try:
        # This dataset usually contains 'data' and 'TCIA_pancreas_labels' folders
        dataset_path = kagglehub.dataset_download("tahsin/pancreasct-dataset")
        print("Path to dataset files:", dataset_path)
    except Exception as e:
        print(f"Error downloading: {e}. Please ensure you have the dataset locally.")
        return

    # Define paths 
    processed_imgs_dir = os.path.join(output_dir, "images")
    processed_masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(processed_imgs_dir, exist_ok=True)
    os.makedirs(processed_masks_dir, exist_ok=True)

    #  CT Windowing Function
    def apply_windowing(image, level=40, width=400):
        """
        Converts raw CT Hounsfield Units (HU) to 0-255 range.
        Standard Soft Tissue Window: Level=40, Width=400.
        """
        lower = level - (width / 2)
        upper = level + (width / 2)
        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)  # Normalize 0 to 1
        return (image * 255).astype(np.uint8)

    # Find and Match 3D Files 
    print("Scanning for 3D volumes (NIfTI)...")
    
    image_volumes = {}
    label_volumes = {}

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                # Extract ID (assuming format like 'PANCREAS_0001' or 'label0001')
                # We strip non-digits to find the ID: '0001'
                file_id = ''.join(filter(str.isdigit, file))
                
                if "label" in file.lower() or "mask" in file.lower():
                    label_volumes[file_id] = full_path
                else:
                    image_volumes[file_id] = full_path

    # Find common IDs
    common_ids = sorted(list(set(image_volumes.keys()) & set(label_volumes.keys())))
    print(f"Found {len(common_ids)} matched 3D subjects.")

    #  Process Volumes: Slice 3D -> 2D 
    sample_registry = [] # Stores (subject_id, slice_filename)

    print("Slicing 3D volumes into 2D images...")
    for subj_id in tqdm(common_ids):
        # Load 3D volumes
        img_nii = nib.load(image_volumes[subj_id])
        lbl_nii = nib.load(label_volumes[subj_id])

        # Get data as numpy arrays
        img_data = nib.as_closest_canonical(img_nii).get_fdata()
        lbl_data = nib.as_closest_canonical(lbl_nii).get_fdata()

        # Iterate through Axial slices 
        num_slices = img_data.shape[2]
        
        for i in range(num_slices):
            slice_img = img_data[:, :, i]
            slice_mask = lbl_data[:, :, i]

            # FILTERING: Only save slices that contain the pancreas?
            has_pancreas = np.sum(slice_mask) > 0
            
            if has_pancreas or random.random() < 0.1: 
                # Resize to standard size (e.g., 512x512 -> 256x256) if needed for Lite model
                # slice_img = cv2.resize(slice_img, (256, 256))
                # slice_mask = cv2.resize(slice_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                # Windowing (Crucial for CT)
                slice_img_uint8 = apply_windowing(slice_img)
                
                # Masks are usually 0 and 1. Convert to 0 and 255.
                slice_mask_uint8 = (slice_mask > 0).astype(np.uint8) * 255

                # Save files
                # Filename format: SubjectID_SliceIndex.png
                fname = f"{subj_id}_{i:04d}.png"
                
                cv2.imwrite(os.path.join(processed_imgs_dir, fname), slice_img_uint8)
                cv2.imwrite(os.path.join(processed_masks_dir, fname), slice_mask_uint8)

                sample_registry.append({
                    "subject": subj_id,
                    "filename": fname
                })

    # Train/Test
    #   split by Subject ID, not by slice, to avoid data leakage.
    random.seed(42)
    random.shuffle(common_ids)
    
    split_idx = int(len(common_ids) * 0.8)
    train_subjects = set(common_ids[:split_idx])
    test_subjects = set(common_ids[split_idx:])

    train_map = {}
    test_map = {}

    print("Generating JSON mappings...")
    for item in sample_registry:
        s_id = item["subject"]
        fname = item["filename"]
        
        img_path = os.path.join(processed_imgs_dir, fname)
        mask_path = os.path.join(processed_masks_dir, fname)

        if s_id in train_subjects:
            # Format: Image -> [List of Masks] (SAM style usually accepts list)
            train_map[img_path] = [mask_path]
        else:
            # Format: Mask -> Image (Label2Image for evaluation)
            test_map[mask_path] = img_path

    # Save JSONs 
    with open('pancreas_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)

    with open('pancreas_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total 2D slices generated: {len(sample_registry)}")
    print(f"Training slices: {len(train_map)} (from {len(train_subjects)} subjects)")
    print(f"Testing slices: {len(test_map)} (from {len(test_subjects)} subjects)")
    print(f"Data saved to: {output_dir}")
