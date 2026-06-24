import os
import re
import shutil

import numpy as np  # Move numpy before cv2
import pandas as pd
import cv2           # Move cv2 after numpy
import imagehash

from PIL import Image
from PIL.ExifTags import TAGS
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def image_data_extractor(df, base_path="../data/raw/images/image_train/"):
    """
    Extract data from product images in given DataFrame with columns 'productid' and 'imageid'.
    Return a new DataFrame with the extracted information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' and 'imageid' columns
    base_path : str
        Path to the folder containing the images
        
    Returns:
    --------
    pandas.DataFrame
        Enhanced DataFrame with image metadata
    """
    
    def process_single_image(row):
        """Process a single image and extract metadata"""
        # Start with ALL the original columns from the input row
        image_data = row.to_dict()  # Copy everything: productid, imageid, prdtypecode
        
        product_id = row["productid"]
        image_id = row["imageid"]
        
        # Construct file path with int conversion to avoid .0 in filename
        file_path = os.path.join(base_path, f"image_{int(image_id)}_product_{int(product_id)}.jpg")
        
        # Check if file exists
        if not os.path.exists(file_path):
            image_data["error"] = "File not found"
            return image_data
            
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Extract basic file properties only (skip RGB analysis as discussed)
            image_data["file_size_kb"] = os.path.getsize(file_path) / 1024
            
        except Exception as e:
            image_data["error"] = str(e)
            
        return image_data
    
    # Process all images
    print(f"Processing {len(df)} images for metadata extraction...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata"):
        result = process_single_image(row)
        results.append(result)
    
    # Create result DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df


def detect_bounding_box_worker(args):
    """
    Worker function to process a single image for bounding box detection using OpenCV.
    
    Parameters:
    -----------
    args : tuple
        (product_id, image_id, input_folder, output_folder, save_images) tuple
        
    Returns:
    --------
    tuple or None
        (product_id, x, y, w, h, ar) if successful, None if failed
    """
    product_id, image_id, input_folder, output_folder, save_images = args
    
    # Construct input path with int conversion
    input_path = os.path.join(
        input_folder, f"image_{int(image_id)}_product_{int(product_id)}.jpg"
    )
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None
    
    try:
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            return None
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            ar = w / h if h > 0 else 0
            
            # Save bounding box image if requested
            if save_images and output_folder:
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(
                    output_folder, f"image_{int(image_id)}_product_{int(product_id)}_bbox.jpg"
                )
                
                # Draw bounding box on image
                bbox_img = img.copy()
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(output_path, bbox_img)
            
            return product_id, x, y, w, h, ar
        else:
            # No contours found, use entire image as bounding box
            print(f"No contours found product {product_id} with 'imageid' {image_id}: using full image dimensions")
            h, w = img.shape[:2]
            ar = w / h if h > 0 else 0
            return product_id, 0, 0, w, h, ar
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def detect_bounding_box_parallel(df, base_path="../data/raw/images/image_train/", save_images=False, n_workers=4):
    """
    Detect bounding boxes for all images in parallel using OpenCV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'productid' and 'imageid' columns
    base_path : str
        Path to input images
    save_images : bool
        Whether to save bounding box images
    n_workers : int
        Number of parallel workers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added bounding box columns
    """
    
    # Prepare columns for bounding box coordinates if they don't exist
    for col in ["bb_x", "bb_y", "bb_w", "bb_h", "bb_ar"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype='float64')  # Don't use None - preserves data types
    
    # Ensure productid and imageid remain as integers - CRITICAL FIX
    if 'productid' in df.columns:
        df['productid'] = df['productid'].astype('int64')
    if 'imageid' in df.columns:
        df['imageid'] = df['imageid'].astype('int64')
    
    input_folder = base_path
    output_folder = base_path.replace("image_train", "image_train_bbox") if save_images else None
    
    # Create tasks list with int conversion to avoid float issues
    tasks = []
    for idx, row in df.iterrows():
        product_id = row["productid"]
        image_id = row["imageid"]
        tasks.append((int(product_id), int(image_id), input_folder, output_folder, save_images))
    
    print(
        f"\033[1mdetect_bounding_box_parallel()\033[0m: Found {len(tasks):,} images to process."
    )
    if save_images:
        print(f"Bounding box images will be saved to: {output_folder}")
    else:
        print("Bounding box images will NOT be saved (coordinates only).")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(detect_bounding_box_worker, tasks),
            total=len(tasks),
            desc="Detecting bounding boxes"
        ))
    
    # Update DataFrame with results using mask-based updates (CRITICAL FIX)
    successful_count = 0
    for result in results:
        if result is not None:
            product_id, x, y, w, h, ar = result
            if x is not None:
                # Use mask instead of product_id as index to avoid row duplication
                mask = df['productid'] == product_id
                df.loc[mask, ["bb_x", "bb_y", "bb_w", "bb_h", "bb_ar"]] = [x, y, w, h, ar]
                successful_count += 1
    
    print(f"Successfully processed {successful_count} of {len(tasks)} images")
    
    return df


def crop_pad_and_resize_image_worker(
    input_path, output_path, bbox=None, target_size=(299, 299), min_length=75
):
    """
    Process a single image by cropping to a bounding box and resizing to target size.

    Parameters:
    -----------
    input_path : str
        Path to the input image file
    output_path : str
        Path to save the processed image
    bbox : tuple
        (x, y, w, h) bounding box coordinates
    target_size : tuple
        (width, height) for final image size
    min_length : int
        Minimum dimension threshold for exclusion

    Returns:
    --------
    tuple
        (product_id, downscale_flag, upscale_flag, exclude_flag)
    """

    downscale_flag = 0
    upscale_flag = 0
    exclude_flag = 0

    # Extract product_id from filename for return value
    filename = os.path.basename(input_path)
    product_id_match = re.search(r"product_(\d+)", filename)
    product_id = int(product_id_match.group(1)) if product_id_match else 0

    try:
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            return product_id, downscale_flag, upscale_flag, 1  # exclude

        # Crop to bounding box if provided
        if bbox and bbox[0] is not None:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Ensure bounding box is within image bounds
            img_h, img_w = img.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            # Crop the image
            img = img[y : y + h, x : x + w]

        # Check if image is too small
        current_h, current_w = img.shape[:2]
        if min(current_w, current_h) < min_length:
            exclude_flag = 1
            return product_id, downscale_flag, upscale_flag, exclude_flag

        # Calculate scaling needed
        target_w, target_h = target_size
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        scale = min(scale_w, scale_h)  # Maintain aspect ratio

        # Determine if upscaling or downscaling
        if scale > 1:
            upscale_flag = 1
        elif scale < 1:
            downscale_flag = 1

        # Resize image maintaining aspect ratio
        new_w = int(current_w * scale)
        new_h = int(current_h * scale)
        img_resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )

        # Create final image with padding to exact target size
        final_img = np.full(
            (target_h, target_w, 3), 255, dtype=np.uint8
        )  # White background

        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        final_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img_resized

        # Save processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_img)

        return product_id, downscale_flag, upscale_flag, exclude_flag

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return product_id, downscale_flag, upscale_flag, 1  # exclude on error


def crop_pad_and_resize_image_parallel(df, base_path="../data/raw/images/image_train/", target_size=(299, 299), min_length=75, n_workers=4):
    """
    Crop, pad, and resize all images in parallel.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with image data and bounding box information
    base_path : str
        Path to input images
    target_size : tuple
        Target size for processed images
    min_length : int
        Minimum dimension for exclusion threshold
    n_workers : int
        Number of parallel workers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added processing flags
    """
    
    input_folder = base_path
    output_folder = "../data/processed/images/image_train_cpr/"
    
    # Prepare processing flag columns with proper data types (FIXED)
    columns_cpr = ["downscaled", "upscaled", "exclude"]
    for c_cpr in columns_cpr:
        if c_cpr not in df.columns:
            df[c_cpr] = pd.Series(0, index=df.index, dtype='int64')  # Use pd.Series instead of just 0
    
    # Preserve ALL existing column data types - CRITICAL FIX we added
    if 'productid' in df.columns:
        df['productid'] = df['productid'].astype('int64')
    if 'imageid' in df.columns:
        df['imageid'] = df['imageid'].astype('int64')
    if 'prdtypecode' in df.columns:
        df['prdtypecode'] = df['prdtypecode'].astype('int64')
    if 'file_size_kb' in df.columns:
        df['file_size_kb'] = df['file_size_kb'].astype('float64')
    # Bounding box columns should remain float64 for NaN compatibility
    for col in ['bb_x', 'bb_y', 'bb_w', 'bb_h', 'bb_ar']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Create tasks list
    tasks = []
    for idx, row in df.iterrows():
        # Get productid and imageid from the DataFrame (FIXED: convert to int immediately)
        product_id = row["productid"]
        image_id = row["imageid"]
        
        # Construct the file paths with int conversion (FIXED: added int() to avoid .0 in filename)
        input_path = os.path.join(
            input_folder, f"image_{int(image_id)}_product_{int(product_id)}.jpg"
        )
        output_path = os.path.join(
            output_folder, f"image_{int(image_id)}_product_{int(product_id)}_cpr.jpg"
        )
        
        # Get bounding box if available
        bbox = None
        if 'bb_x' in df.columns and pd.notna(row['bb_x']):
            bbox = (row['bb_x'], row['bb_y'], row['bb_w'], row['bb_h'])
        
        if os.path.exists(input_path):
            tasks.append((input_path, output_path, bbox, target_size, min_length))
    
    print(f"\033[1mcrop_pad_and_resize_image_parallel()\033[0m: Found {len(tasks):,} images to process.")
    if len(df) - len(tasks) > 0:
        print(f"Warning: {len(df) - len(tasks)} images referenced in DataFrame were not found on disk.")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            [crop_pad_and_resize_image_worker(*task) for task in tasks],
            total=len(tasks),
            desc="Cropping & resizing images"
        ))
    
    # Update DataFrame with results using mask-based updates (CRITICAL FIX we added)
    downscaled_count = 0
    upscaled_count = 0
    excluded_count = 0
    
    for result in results:
        product_id, downscale_flag, upscale_flag, exclude_flag = result
        
        # FIXED: Use mask instead of product_id as index to avoid row duplication
        mask = df['productid'] == product_id
        
        if downscale_flag == 1:
            df.loc[mask, "downscaled"] = 1
            downscaled_count += 1
            
        if upscale_flag == 1:
            df.loc[mask, "upscaled"] = 1
            upscaled_count += 1
            
        if exclude_flag == 1:
            df.loc[mask, "exclude"] = 1
            excluded_count += 1
    
    print(f"Successfully processed {len(results)} of {len(tasks)} images:")
    print(f"    - Downscaled {downscaled_count} images")
    print(f"    - Upscaled {upscaled_count} images")
    print(f"    - Flagged {excluded_count} images for potential exclusion due to small size")
    
    return df


def copy_image_to_class_folders(df_image_train, input_folder="../data/processed/images/image_train_cpr/", 
                               output_folder="../data/processed/images/image_train_vgg16/", cleanup_intermediate=False):
    """
    Copy processed images into class-specific folders for PyTorch training.
    
    Parameters:
    -----------
    df_image_train : pandas.DataFrame
        DataFrame with image metadata and category information
    input_folder : str
        Path to processed images (_cpr folder)
    output_folder : str
        Path to create class folder structure
    cleanup_intermediate : bool
        Whether to delete intermediate _cpr folder after completion
    """
    
    # Ensure proper data types - CRITICAL FIX for proper folder naming
    if 'prdtypecode' in df_image_train.columns:
        df_image_train['prdtypecode'] = df_image_train['prdtypecode'].astype('int64')
    if 'productid' in df_image_train.columns:
        df_image_train['productid'] = df_image_train['productid'].astype('int64')
    if 'imageid' in df_image_train.columns:
        df_image_train['imageid'] = df_image_train['imageid'].astype('int64')
    
    # Create train and validation directories
    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get unique classes
    classes = df_image_train['prdtypecode'].unique()
    
    for class_id in classes:
        # Filter images for this class
        class_images = df_image_train[df_image_train['prdtypecode'] == class_id]
        
        # Skip classes with too few samples for train/test split
        if len(class_images) < 2:
            print(f"Skipping class {class_id}: only {len(class_images)} sample(s)")
            continue
        
        # Create class directories
        class_train_dir = os.path.join(train_dir, str(int(class_id)))  # Ensure integer folder names
        class_val_dir = os.path.join(val_dir, str(int(class_id)))
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)
        
        # Split into train/validation (80/20)
        if len(class_images) == 1:
            # Only one image, put in training
            train_images = class_images
            val_images = pd.DataFrame()
        else:
            train_images, val_images = train_test_split(
                class_images, test_size=0.2, random_state=42
            )
        
        # Copy training images
        for idx, row in train_images.iterrows():
            product_id = int(row["productid"])  # Ensure int conversion
            image_id = int(row["imageid"])      # Ensure int conversion
            
            source_path = os.path.join(input_folder, f"image_{image_id}_product_{product_id}_cpr.jpg")
            dest_path = os.path.join(class_train_dir, f"image_{image_id}_product_{product_id}_cpr.jpg")
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
        
        # Copy validation images
        for idx, row in val_images.iterrows():
            product_id = int(row["productid"])  # Ensure int conversion
            image_id = int(row["imageid"])      # Ensure int conversion
            
            source_path = os.path.join(input_folder, f"image_{image_id}_product_{product_id}_cpr.jpg")
            dest_path = os.path.join(class_val_dir, f"image_{image_id}_product_{product_id}_cpr.jpg")
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
        
        print(f"Processed class {int(class_id)}: ")
        print(f"    {len(train_images)} training images ")
        print(f"    {len(val_images)} validation images")
    
    # Cleanup intermediate folder if requested
    if cleanup_intermediate and os.path.exists(input_folder):
        print(f"Cleaning up intermediate folder: {input_folder}")
        shutil.rmtree(input_folder)
    
    print("Class folder organization completed!")


def hash_parallel(df, base_path="../data/processed/images/image_train_cpr/", hash_size=8, n_workers=4):
    """
    Generate perceptual hashes for all images in parallel.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with image data
    base_path : str
        Path to images
    hash_size : int
        Size of perceptual hash
    n_workers : int
        Number of parallel workers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added hash column
    """
    
    def hash_worker(args):
        """Worker function for hashing"""
        product_id, image_id, folder_path, hash_size = args
        
        image_path = os.path.join(folder_path, f"image_{int(image_id)}_product_{int(product_id)}_cpr.jpg")
        
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                phash = str(imagehash.phash(img, hash_size=hash_size))
                return product_id, phash
            else:
                return product_id, None
        except Exception as e:
            print(f"Error hashing {image_path}: {e}")
            return product_id, None
    
    # Add hash column if it doesn't exist
    if 'phash' not in df.columns:
        df['phash'] = pd.Series(dtype='object')
    
    # Create tasks list
    tasks = [(row["productid"], row["imageid"], base_path, hash_size) for idx, row in df.iterrows()]
    
    print(f"Generating perceptual hashes for {len(tasks):,} images...")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(hash_worker, tasks),
            total=len(tasks),
            desc="Generating hashes"
        ))
    
    # Update DataFrame with results
    for result in results:
        product_id, phash = result
        if phash is not None:
            mask = df['productid'] == product_id
            df.loc[mask, 'phash'] = phash
    
    return df


def find_duplicates_parallel(df, threshold=0):
    """
    Find duplicate images based on perceptual hash comparison.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'phash' column
    threshold : int
        Hamming distance threshold for duplicates (0 = exact match)
        
    Returns:
    --------
    tuple
        (updated_dataframe, unique_duplicates_dataframe)
    """
    
    print(f"Finding duplicates with threshold {threshold}...")
    
    # Add duplicate flag column
    if 'phash_duplicate' not in df.columns:
        df['phash_duplicate'] = 0
    
    # Find duplicates
    duplicates = []
    phash_groups = df.dropna(subset=['phash']).groupby('phash')
    
    for phash, group in phash_groups:
        if len(group) > 1:
            # Mark all but the first as duplicates
            duplicate_indices = group.index[1:]
            df.loc[duplicate_indices, 'phash_duplicate'] = 1
            duplicates.extend(duplicate_indices.tolist())
    
    print(f"Found {len(duplicates)} duplicate images")
    
    # Create unique duplicates DataFrame
    unique_duplicates = df.loc[duplicates].copy()
    
    return df, unique_duplicates

def display_phash(hash_hex, size=8, scale=32):
    """
    Visualize a product image's perceptual hash as a black and white image similar to a QR code.

    Parameters:
    - hash_hex: The perceptual hash value as a hexadecimal string
    - size: Width/height of the hash visualization (default 8 for 64-bit hash)
    - scale: Scaling factor to make the visualization larger

    Returns:
    - PIL Image object
    """
    # Convert hex hash to binary
    binary = bin(int(hash_hex, 16))[2:].zfill(size * size)

    # Create a new blank image (mode '1' is 1-bit pixels, black and white)
    img = Image.new("1", (size, size))

    # Fill the image based on the hash bits
    for i in range(size):
        for j in range(size):
            pixel_index = i * size + j
            if pixel_index < len(binary):
                # Set pixel to white (1) if the hash bit is 1
                img.putpixel((j, i), int(binary[pixel_index]))

    # Scale it up for better visibility
    img = img.resize((size * scale, size * scale), Image.NEAREST)

    return img

def preprocess_image(image_array):
    """
    Process an image through the complete preprocessing pipeline.
    
    Parameters:
    - image_array: numpy array of the input image
    
    Returns:
    - img_with_bb: Image with bounding box drawn
    - canvas: Cropped, padded and resized image 
    - phash_img: Perceptual hash visualization
    - df_image_data: DataFrame with processing metadata
    """
    import numpy as np
    import pandas as pd
    import imagehash
    from PIL import Image
    
    # Process the image through all preprocessing steps
    # 1. Create a copy for bounding box detection
    img_bb = image_array.copy()

    # Apply bounding box detection
    gray = cv2.cvtColor(img_bb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 247, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize coordinates
    x, y, w, h = 0, 0, image_array.shape[1], image_array.shape[0]

    # If contours found, get the largest one
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw bounding box on the image
    img_with_bb = image_array.copy()
    img_with_bb = cv2.cvtColor(img_with_bb, cv2.COLOR_RGB2BGR)
    img_with_bb = cv2.rectangle(img_with_bb, (x, y), (x + w, y + h), (0, 0, 255), 16)
    img_with_bb = cv2.cvtColor(img_with_bb, cv2.COLOR_BGR2RGB)

    # 2. Crop, pad and resize
    # Calculate aspect ratio
    ar = w / h if h > 0 else 1

    # Crop to bounding box
    img_cropped = image_array[y : y + h, x : x + w]

    # Define target size
    target_size = (224, 224)  # VGG16: 224

    # Calculate new dimensions while preserving aspect ratio
    target_w, target_h = target_size

    # Resize based on the smaller dimension to preserve aspect ratio
    if w >= h:  # Wider than tall
        new_w = round(target_h * ar)
        new_h = target_h
    else:  # Taller than wide
        new_w = target_w
        new_h = round(target_w / ar)

    # Ensure the longer side doesn't exceed the target size
    if new_w > target_w:
        new_w = target_w
        new_h = round(target_w / ar)
    if new_h > target_h:
        new_h = target_h
        new_w = round(target_h * ar)

    # Determine interpolation method
    min_dim = min(h, w)
    downscaled = 1 if min_dim > min(target_size) else 0
    upscaled = 1 if min_dim < min(target_size) else 0
    exclude = 1 if min_dim < 75 else 0

    interpolation = (
        cv2.INTER_AREA
        if downscaled
        else cv2.INTER_CUBIC if upscaled else cv2.INTER_LINEAR
    )

    # Resize the image while maintaining aspect ratio
    resized_img = cv2.resize(img_cropped, (new_w, new_h), interpolation=interpolation)

    # Create a blank white canvas of the target size
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    # Calculate positioning to center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # Place the resized image onto the canvas
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

    # 3. Calculate perceptual hash
    img_pil = Image.fromarray(image_array)
    hash_value = str(imagehash.phash(img_pil, hash_size=8))

    # Generate hash visualization
    phash_img = display_phash(hash_value, size=8, scale=32)

    # Create a dataframe with metadata
    image_data = {
        "name": "Uploaded image",
        "file_size_kb": [len(image_array) / 1024],
        "mean_r": [np.mean(image_array[:, :, 0])],
        "mean_g": [np.mean(image_array[:, :, 1])],
        "mean_b": [np.mean(image_array[:, :, 2])],
        "bb_x": [x],
        "bb_y": [y],
        "bb_w": [w],
        "bb_h": [h],
        "bb_ar": [ar],
        "downscale": [downscaled],
        "upscale": [upscaled],
        "exclude": [exclude],
        "phash": [hash_value],
    }

    df_image_data = pd.DataFrame(image_data, index=["name"])

    return img_with_bb, canvas, phash_img, df_image_data