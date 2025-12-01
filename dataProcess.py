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

    

    # 1. Define your paths
    # REPLACE THIS with your actual path
    kits_dataset_path = ".." + path + "/AUGMENTED/DATASET_FINAL/"

    base_dir = kits_dataset_path
    images_dir = os.path.join(base_dir, "JPEGImages")
    masks_dir = os.path.join(base_dir, "Annotations")

    # 2. Get list of filenames
    # Filter for .jpg in images and .png in masks
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
    all_mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

    print(f"Total images: {len(all_image_files)}")
    print(f"Total masks: {len(all_mask_files)}")

    # # 3. Match Images to Masks
    # Logic: Group all masks belonging to a specific image
    # matched_samples will be a list of tuples: (img_file, [mask_file_1, mask_file_2, ...])
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


    # 4. Shuffle the data for a random split
    # We shuffle the IMAGES (samples), not the individual pairs, to prevent data leakage.
    random.seed(42)
    random.shuffle(matched_samples)

    # 5. Calculate split index (80% for training)
    split_index = int(len(matched_samples) * 0.8)

    train_samples = matched_samples[:split_index]
    test_samples = matched_samples[split_index:]

    # 6. Create the dictionaries
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
    # Here we flatten the relationship. Each mask is a unique key pointing to its image.
    for img_file, mask_list in test_samples:
        img_path = os.path.join(images_dir, img_file)

        for mask_file in mask_list:
            mask_path = os.path.join(masks_dir, mask_file)
            test_map[mask_path] = img_path

    # 7. Save to JSON files
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

    # 1. Define your paths
    # Assuming the script is running in the parent directory of 'Lung Segmentation'
    base_dir = path + "/Lung Segmentation" 
    images_dir = os.path.join(base_dir, "CXR_png")
    masks_dir = os.path.join(base_dir, "masks")

    # 2. Get list of filenames
    # CHANGED: 'CXR_png' suggests images are .png, not .jpg
    all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
    all_mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

    print(f"Total images found: {len(all_image_files)}")
    print(f"Total masks found: {len(all_mask_files)}")

    # 3. Match Images to Masks
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

    # 4. Shuffle the data for a random split
    random.seed(42)
    random.shuffle(matched_samples)

    # 5. Calculate split index (80% for training)
    split_index = int(len(matched_samples) * 0.8)

    train_samples = matched_samples[:split_index]
    test_samples = matched_samples[split_index:]

    # 6. Create the dictionaries
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

    # 7. Save to JSON files
    with open('lung_xray_image2label_train.json', 'w') as f:
        json.dump(train_map, f, indent=4)

    with open('lung_xray_label2image_test.json', 'w') as f:
        json.dump(test_map, f, indent=4)

    print("-" * 30)
    print(f"Total valid images found: {len(matched_samples)}")
    print(f"Training images: {len(train_map)} saved to 'lung_image2label_train.json'")
    print(f"Testing masks: {len(test_map)} saved to 'lung_label2image_test.json'")