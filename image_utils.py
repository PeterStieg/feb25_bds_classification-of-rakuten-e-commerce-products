import os
import cv2
import re
import imagehash

import numpy as np
import pandas as pd

from PIL import Image
from PIL.ExifTags import TAGS
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

"""
Image utility functions for extracting data from images.

This module provides functions for tasks such as:
- Data extraction, e. g. size, format, metadata

Planned features:
- Image processing
- Image classification

These functions are designed to prepare image data for machine learning.
"""


def image_data_extractor(df, base_path="./images/image_train/"):
    """
    Extract data from product images in given DataFrame with columns 'productid' and 'imageid'.
    Return a new DataFrame with the extracted information.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' and 'imageid' columns
    base_path : str, optional
        The base directory path where images are stored, e.g. './images/image_train/'
        Default is './images/image_train/'.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with 'productid', 'imageid', and columns for each extracted data point
    """

    # List to store extracted data for each image
    extracted_data = []

    # Process each row in the input DataFrame
    for _, row in df.iterrows():
        # Get productid and imageid from the DataFrame
        product_id = row["productid"]
        image_id = row["imageid"]

        # Initialize a dictionary to store data for this image
        image_data = {"productid": product_id, "imageid": image_id}

        # Construct the file path
        file_path = f"{base_path}image_{image_id}_product_{product_id}.jpg"

        # Check if file exists
        if not os.path.exists(file_path):
            image_data["error"] = "File not found"  # f"File not found": {file_path}"
            extracted_data.append(image_data)
            continue

        try:
            # Open the image
            img = Image.open(file_path)

            # Extract basic image properties
            image_data["width"] = img.size[0]
            image_data["height"] = img.size[1]
            image_data["format"] = img.format
            image_data["mode"] = img.mode  # RGB, L, etc.
            image_data["file_size_kb"] = os.path.getsize(file_path) / 1024
            image_data["aspect_ratio"] = (
                img.size[0] / img.size[1] if img.size[1] > 0 else None
            )

            # Extract metadata from image
            for key, value in img.info.items():
                # Handle binary data and complex objects
                if isinstance(value, (bytes, bytearray)):
                    continue  # Skip binary data
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # Only include simple data types in the DataFrame
                    image_data[f"meta_{key}"] = value

            # Extract EXIF data
            if hasattr(img, "_getexif") and img._getexif():
                for tag_id, value in img._getexif().items():
                    tag_name = TAGS.get(tag_id, f"tag_{tag_id}")
                    # Skip binary data and complex objects
                    if isinstance(value, (bytes, bytearray)):
                        continue
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        image_data[f"exif_{tag_name}"] = value

            # Color analysis if the image is in RGB mode
            if img.mode == "RGB":
                img_array = np.array(img)

                # Calculate mean RGB values
                mean_rgb = img_array.mean(axis=(0, 1))
                image_data["mean_r"] = mean_rgb[0]
                image_data["mean_g"] = mean_rgb[1]
                image_data["mean_b"] = mean_rgb[2]

                # Calculate dominant brightness
                brightness = (
                    0.299 * img_array[:, :, 0]
                    + 0.587 * img_array[:, :, 1]
                    + 0.114 * img_array[:, :, 2]
                )
                image_data["mean_brightness"] = brightness.mean()

                # Calculate color variance as a simple measure of image complexity
                std_rgb = img_array.std(axis=(0, 1))
                image_data["std_r"] = std_rgb[0]
                image_data["std_g"] = std_rgb[1]
                image_data["std_b"] = std_rgb[2]

        except Exception as e:
            image_data["error"] = str(e)

        finally:
            # Ensure we close the image file
            if "img" in locals() and hasattr(img, "close"):
                img.close()

        # Add this image's data to our list
        extracted_data.append(image_data)

    # Convert the list of dictionaries to a DataFrame
    df_image_data_extractor = pd.DataFrame(extracted_data)

    return df_image_data_extractor


def detect_bounding_box_worker(args):
    """
    Worker function to process a single image for bounding box detection using OpenCV.

    Parameters:
    -----------
        args : tuple
        (product_id, image_id, input_folder, output_folder) tuple

    Returns:
    --------
    tuple
        (product_id, x, y, w, h, ar) coordinates and aspect ratio of bounding box
    """

    product_id, image_id, input_folder, output_folder = args

    try:
        # Construct the file paths
        input_path = os.path.join(
            input_folder, f"image_{image_id}_product_{product_id}.jpg"
        )
        output_path = os.path.join(
            output_folder, f"image_{image_id}_product_{product_id}_bb.jpg"
        )

        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return product_id, None, None, None, None, None

        # Load the image
        img = cv2.imread(input_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary Inverse with adjusted threshold
        _, thresh = cv2.threshold(blurred, 247, 255, cv2.THRESH_BINARY_INV)

        # Try to find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # If contours found proceed
        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate aspect ratio of bounding box
            ar = w / h

            # Draw the bounding box on the original image
            result = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save the result
            cv2.imwrite(output_path, result)

            # Return coordinates, width, height and aspect ratio of bounding box
            return product_id, x, y, w, h, ar

        else:
            # If no contours found, use entire product image as bounding box
            h, w = img.shape[:2]

            # Calculate aspect ratio with zero check
            ar = w / h if h > 0 else 0

            # Save the original image since we couldn't find a bounding box
            cv2.imwrite(output_path, img)

            print(
                f"No contours found product {product_id} with 'imageid' {image_id}: using full image dimensions"
            )

            # Return coordinates, width, height and aspect ratio of image
            return product_id, 0, 0, w, h, ar

    except Exception as e:
        print(f"Error processing product {product_id} with 'imageid' {image_id}: {e}")
        return product_id, None, None, None, None, None


def detect_bounding_box_parallel(df, base_path="./images/image_train/", n_workers=None):
    """
    Detect bounding boxes of images listed in a DataFrame using OpenCV in parallel

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' in index and column for 'imageid'
    base_path : str, optional
        Directory where images are stored, default is './images/image_train/'
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional columns for bounding box coordinates
    """

    # Determine number of workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count()

    # Add suffix _bb to the base path as in "_bounding_box"
    input_folder = base_path
    output_folder = re.sub(r"\/([^\/]*)$", r"_bb/\1", input_folder)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Prepare columns for bounding box coordinates if they don't exist
    for col in ["bb_x", "bb_y", "bb_w", "bb_h", "bb_ar"]:
        if col not in df.columns:
            df[col] = None

    # Create a list of tasks for ProcessPoolExecutor
    tasks = []

    for idx, row in df.iterrows():
        product_id = idx
        image_id = row["imageid"]
        tasks.append((product_id, image_id, input_folder, output_folder))

    print(
        f"\033[1mdetect_bounding_box_parallel()\033[0m: Found {len(tasks):,} images to process."
    )

    # Process bounding box detection in parallel

    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Map the worker function to all tasks (more efficient than submit)
        future_results = list(executor.map(detect_bounding_box_worker, tasks))

        # Process results with a progress bar
        for result in tqdm(
            future_results, total=len(tasks), desc="Detecting bounding boxes"
        ):
            results.append(result)

    # Update DataFrame with results
    successful = 0
    for result in results:
        if result is not None:  # Check if result exists
            product_id, x, y, w, h, ar = result
            if x is not None:  # Check if processing was successful
                df.loc[product_id, ["bb_x", "bb_y", "bb_w", "bb_h", "bb_ar"]] = [
                    x,
                    y,
                    w,
                    h,
                    ar,
                ]
                successful += 1

    print(f"Successfully processed {successful} of {len(tasks)} images")

    return df


def crop_pad_and_resize_image_worker(
    input_path, output_path, bbox=None, target_size=(299, 299), min_length=75
):
    """
    Process a single image by cropping to a bounding box and resizing to target size.

    Parameters:
    -----------
    input_path : str
        Path to the input image file.
    output_path : str
        Path where the processed image will be saved.
    bbox : tuple, optional
        Bounding box coordinates as (x, y, width, height). If None, uses full image.
    target_size : tuple, optional
        Target size for resizing, default is (299, 299).
    min_length : int, optional
        Minimum acceptable dimension size for quality filtering, default is 75.

    Returns:
    -------
    tuple
        (downscaled_flag, upscaled_flag, exclude_flag, success_flag) where:
        - downscaled_flag: 1 if image was downscaled, 0 otherwise
        - upscaled_flag: 1 if image was upscaled, 0 otherwise
        - exclude_flag: 1 if image should be excluded due to small size, 0 otherwise
        - success_flag: True if processing succeeded, False otherwise
    """

    # Set image processing flags to zero
    downscaled = 0
    upscaled = 0
    exclude = 0

    try:
        # Read the image
        img = cv2.imread(input_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return downscaled, upscaled, exclude, False

        # Crop to bounding box if provided
        if bbox and all(v is not None for v in bbox):
            x, y, w, h = bbox
            img = img[y : y + h, x : x + w]

        # Get current dimensions
        h, w = img.shape[:2]
        min_dim = min(h, w)
        max_dim = max(h, w)
        aspect_ratio = w / h if h > 0 else 1

        # Determine if we need to upscale or downscale
        # We'll use the smallest dimension for comparison since we'll be preserving
        if min_dim > min(target_size):
            # Downscaling - use INTER_AREA for better quality
            interpolation = cv2.INTER_AREA
            downscaled = 1

        elif min_dim < min(target_size):
            # Upscaling - use INTER_CUBIC for better quality
            interpolation = cv2.INTER_CUBIC
            upscaled = 1

            # Set flag if image's shortest side is smaller than min_length
            if min_dim < min_length:
                exclude = 1
        else:
            # Edge case: Already the right size in at least one dimension
            interpolation = cv2.INTER_LINEAR  # Compromise for mixed cases

        # Calculate new dimensions while preserving aspect ratio
        target_w, target_h = target_size

        # Resize based on the smaller dimension to preserve aspect ratio
        if w >= h:  # Wider than tall
            new_w = round(target_h * aspect_ratio)
            new_h = target_h
        else:  # Taller than wide
            new_w = target_w
            new_h = round(target_w / aspect_ratio)

        # Ensure the longer side doesn't exceed the target size
        if new_w > target_w:
            new_w = target_w
            new_h = round(target_w / aspect_ratio)
        if new_h > target_h:
            new_h = target_h
            new_w = round(target_h * aspect_ratio)

        # Resize the image while maintaining aspect ratio
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # Create a blank white canvas of the target size
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        # Calculate positioning to center the image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        # Place the resized image onto the canvas
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

        # Save the processed image
        cv2.imwrite(output_path, canvas)

        return downscaled, upscaled, exclude, True  # Note: Success flag is True here

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return downscaled, upscaled, exclude, False


def worker_wrapper_cpr(args):
    """
    Wrapper function that unpacks arguments for parallel image cropping and resizing tasks.

    Parameters:
    -----------
    args : tuple
        A tuple containing (input_path, output_path, bbox, target_size, min_length)

    Returns:
    --------
    tuple
        (downscale_flag, upscale_flag, exclude_flag, success_flag) from the crop_pad_and_resize_image_worker function
    """
    input_path, output_path, bbox, target_size, min_length = args
    return crop_pad_and_resize_image_worker(
        input_path, output_path, bbox, target_size, min_length
    )


def crop_pad_and_resize_image_parallel(
    df,
    base_path="./images/image_train/",
    target_size=(299, 299),
    min_length=75,
    n_workers=None,
):
    """
    Crop and resize images contained in a DataFrame according to bounding box dimensions.
    Add columns 'downscaled', 'upscaled' and 'exclude' to DataFrame

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' in index and columns for 'imageid', 'bb_x', 'bb_y', 'bb_w', 'bb_h'
    base_path : str, optional
        Directory where images are stored
        Default is './images/image_train/'
    target_size : tuple, optional
        Target size for resized images
        Default is (299, 299)
    min_length : int, optional
        Minimum acceptable dimension size for quality filtering
        Default is 75
    n_workers : int, optional
        Number of worker processes to use.
        If None, uses all available CPU cores.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with the new 'downscaled', 'upscaled', and 'exclude' columns
    """

    # Determine number of workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count()

    # Add suffix _cpr to the base path for "crop, pad and resize"
    input_folder = base_path
    output_folder = re.sub(r"\/([^\/]*)$", r"_cpr/\1", input_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a list of tasks for ProcessPoolExecutor
    worker_args = []

    # Prepare to track type of image processing in the DataFrame
    columns_cpr = ["downscaled", "upscaled", "exclude"]

    for c_cpr in columns_cpr:
        if c_cpr not in df.columns:
            df[c_cpr] = 0

    # Check if bounding box columns exist in DataFrame
    has_bbox = all(col in df.columns for col in ["bb_x", "bb_y", "bb_w", "bb_h"])

    # Track images that couldn't be found
    missing_images = 0

    # Create a mapping from task index to product ID for later reference
    task_to_product = {}

    # Process each row in the input DataFrame
    task_idx = 0
    for idx, row in df.iterrows():
        # Get productid and imageid from the DataFrame
        product_id = idx
        image_id = row["imageid"]

        # Construct the file paths
        input_path = os.path.join(
            input_folder, f"image_{image_id}_product_{product_id}.jpg"
        )

        output_path = os.path.join(
            output_folder, f"image_{image_id}_product_{product_id}_cpr.jpg"
        )

        # Get bounding box if available
        bbox = None
        if has_bbox:
            bbox = (row.get("bb_x"), row.get("bb_y"), row.get("bb_w"), row.get("bb_h"))

        # Only add if the input file exists
        if os.path.exists(input_path):
            worker_args.append((input_path, output_path, bbox, target_size, min_length))
            task_to_product[task_idx] = product_id
            task_idx += 1
        else:
            missing_images += 1

    print(
        f"\033[1mcrop_pad_and_resize_image_parallel()\033[0m: Found {len(worker_args):,} images to process."
    )

    if missing_images > 0:
        print(
            f"Warning: {missing_images} images referenced in DataFrame were not found on disk."
        )

    # Process images in parallel
    downscale_flags = []
    upscale_flags = []
    exclude_flags = []
    success_flags = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process results with a progress bar
        results = list(
            tqdm(
                executor.map(worker_wrapper_cpr, worker_args),
                total=len(worker_args),
                desc="Cropping & resizing images",
            )
        )

        # Process the results and update DataFrame
        for i, (downscale_flag, upscale_flag, exclude_flag, success_flag) in enumerate(
            results
        ):
            downscale_flags.append(downscale_flag)
            upscale_flags.append(upscale_flag)
            exclude_flags.append(exclude_flag)
            success_flags.append(success_flag)

            # Only update the DataFrame if processing was successful
            if success_flag:
                product_id = task_to_product[i]

                # Update all flags independently (not mutually exclusive)
                if downscale_flag == 1:
                    df.loc[product_id, "downscaled"] = 1

                if upscale_flag == 1:
                    df.loc[product_id, "upscaled"] = 1

                if exclude_flag == 1:
                    df.loc[product_id, "exclude"] = 1

    # Count successful operations and categorized images
    successful = sum(success_flags)
    downscaled = sum(downscale_flags)
    upscaled = sum(upscale_flags)
    excluded = sum(exclude_flags)

    print(f"Successfully processed {successful} of {len(worker_args)} images:")
    print(f"  - Downscaled {downscaled} images")
    print(f"  - Upscaled {upscaled} images")
    print(f"  - Flagged {excluded} images for potential exclusion due to small size")

    # Return the DataFrame with the new columns
    return df


def hash_worker(args):
    """
    Worker function to compute perceptual hash for a single image.

    Parameters:
    -----------
    args : tuple
        (product_id, image_id, input_folder, hash_size) tuple

    Returns:
    --------
    tuple
        (product_id, image_id, hash_value) where hash_value is the computed perceptual hash
    """

    product_id, image_id, input_folder, hash_size = args

    try:
        # Construct the file path
        input_path = os.path.join(
            input_folder, f"image_{image_id}_product_{product_id}.jpg"
        )

        # Check if file exists
        if not os.path.exists(input_path):
            return product_id, image_id, None

        # Open the image and compute perceptual hash

        with Image.open(input_path) as img:
            # Using perceptual hash as it's best for product images
            hash_value = str(imagehash.phash(img, hash_size=hash_size))

        return product_id, image_id, hash_value

    except Exception as e:
        print(f"Error processing product {product_id}, image {image_id}: {e}")
        return product_id, image_id, None


def hash_parallel(df, base_path="./images/image_train/", hash_size=8, n_workers=None):
    """
    Compute perceptual hashes for images in DataFrame using parallel processing.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' in index and column for 'imageid'
    base_path : str, optional
        Directory where images are stored, default is './images/image_train/'
    hash_size : int, optional
        Size of the hash (8 creates a 64-bit hash, 16 creates a 256-bit hash)
        Default is 8
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional column 'phash' for images' perceptual hash values
    """

    # Set up workers and hash column
    n_workers = os.cpu_count() if n_workers is None else n_workers
    if "phash" not in df.columns:
        df["phash"] = None

    # Create tasks list
    tasks = [(idx, row["imageid"], base_path, hash_size) for idx, row in df.iterrows()]

    print(
        f"\033[1mhash_parallel()\033[0m: Computing perceptual hashes for {len(tasks):,} images."
    )

    # Process hashing in parallel
    successful = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process with progress bar
        for product_id, image_id, hash_value in tqdm(
            executor.map(hash_worker, tasks),
            total=len(tasks),
            desc="Computing perceptual hashes",
        ):
            # Update DataFrame if hash was computed successfully
            if hash_value is not None:
                df.loc[product_id, "phash"] = hash_value
                successful += 1

    print(f"Successfully computed hashes for {successful} of {len(tasks)} images")

    return df


def process_hash_chunk(chunk_data):
    """
    Process a chunk of product IDs to find duplicates.

    Parameters:
    -----------
    chunk_data : tuple
        (product_id_chunk, hash_dict, threshold) where:
        - product_id_chunk is a list of product IDs to process
        - hash_dict is a dictionary mapping product IDs to hash objects
        - threshold is the maximum bit difference to consider as duplicate

    Returns:
    --------
    tuple
        (local_duplicates, local_duplicate_flags) where:
        - local_duplicates is a list of (id1, id2, distance) tuples
        - local_duplicate_flags is a dict mapping duplicate IDs to 1
    """
    product_id_chunk, hash_dict, threshold = chunk_data
    local_duplicates = []
    local_duplicate_flags = {}

    # For each product in this chunk
    for idx in product_id_chunk:
        hash_obj = hash_dict[idx]

        # Compare against all other products
        for other_idx, other_hash in hash_dict.items():
            # Skip self-comparison
            if idx == other_idx:
                continue

            # Calculate hash difference
            distance = hash_obj - other_hash

            # If distance is within threshold, mark as duplicate
            if distance <= threshold:
                local_duplicates.append((idx, other_idx, distance))
                local_duplicate_flags[other_idx] = 1

    return local_duplicates, local_duplicate_flags


def find_duplicates_parallel(df, threshold=5, n_workers=None):
    """
    Find duplicate images based on perceptual hash similarity in parallel.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'phash' column with perceptual hash values
    threshold : int, optional
        Maximum bit difference to consider as duplicate, default is 5
    n_workers : int, optional
        Number of worker processes to use. If None, uses all available CPU cores.

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with an additional 'phash_duplicate' column
    list
        List of tuples containing (product_id1, product_id2, distance) for duplicate pairs
    """
    # Set number of workers
    n_workers = os.cpu_count() if n_workers is None else n_workers

    # Create 'phash_duplicate' column if it doesn't exist
    if "phash_duplicate" not in df.columns:
        df["phash_duplicate"] = 0

    # Filter out rows without hash values
    df_with_hash = df.dropna(subset=["phash"]).copy()

    # Convert hash strings to hash objects (do this once)
    hash_dict = {}
    for idx, row in df_with_hash.iterrows():
        hash_dict[idx] = imagehash.hex_to_hash(row["phash"])

    # Get all product IDs with valid hashes
    all_product_ids = list(hash_dict.keys())

    # Split product IDs into chunks for parallel processing
    # Use smaller chunks to ensure even distribution of work
    chunk_size = max(1, len(all_product_ids) // (n_workers * 10))
    chunks = [
        all_product_ids[i : i + chunk_size]
        for i in range(0, len(all_product_ids), chunk_size)
    ]

    print(
        f"Processing {len(all_product_ids):,} images in {len(chunks)} chunks using {n_workers} workers"
    )

    # Prepare chunk data for workers
    chunk_data = [(chunk, hash_dict, threshold) for chunk in chunks]

    # Process chunks in parallel
    all_duplicates = []
    all_duplicate_flags = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process results with progress bar
        results = list(
            tqdm(
                executor.map(process_hash_chunk, chunk_data),
                total=len(chunks),
                desc="Finding duplicate hashes",
            )
        )

        # Merge results
        for local_duplicates, local_duplicate_flags in results:
            all_duplicates.extend(local_duplicates)
            all_duplicate_flags.update(local_duplicate_flags)

    # Update the DataFrame with duplicate flags
    for idx in all_duplicate_flags:
        df.loc[idx, "phash_duplicate"] = 1

    # Remove any duplicate pairs (can happen with chunked processing)
    # We sort the tuple elements to ensure consistent ordering
    seen_pairs = set()
    unique_duplicates = []

    for a, b, c in all_duplicates:
        # Ensure consistent ordering to detect duplicates
        pair = (min(a, b), max(a, b))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_duplicates.append((pair[0], pair[1], c))

    print(
        f"Found {len(unique_duplicates)} unique duplicate pairs with threshold {threshold}"
    )
    print(f"Marked {len(all_duplicate_flags)} images as duplicates in the DataFrame")

    return df, unique_duplicates


def create_duplicates_dataframe(unique_duplicates):
    """
    Convert a list of duplicate image pairs into a structured DataFrame.

    Parameters:
    -----------
    unique_duplicates : list
        List of tuples containing (product_id1, product_id2, distance)
        where product_id1 and product_id2 are the IDs of similar images
        and distance is their perceptual hash difference

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns for original_id, duplicate_id, and distance
    """
    # Create a dictionary with meaningful column names
    data = {
        "Original ID": [pair[0] for pair in unique_duplicates],
        "Duplicate ID": [pair[1] for pair in unique_duplicates],
        "hash distance": [pair[2] for pair in unique_duplicates],
    }

    # Create the DataFrame
    duplicates_df = pd.DataFrame(data)

    # Sort by hash distance (most similar pairs first)
    duplicates_df = duplicates_df.sort_values("hash distance")

    # Add a similarity percentage column (100% - normalized distance)
    # Assuming 64-bit hash (hash_size=8), max possible distance is 64
    duplicates_df["Similarity, pct"] = 100 * (1 - duplicates_df["hash distance"] / 64)

    return duplicates_df
