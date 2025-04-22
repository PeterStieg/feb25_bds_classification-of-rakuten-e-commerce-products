import os
import cv2
import re

import numpy as np
import pandas as pd

from PIL import Image
from PIL.ExifTags import TAGS
from concurrent.futures import ProcessPoolExecutor
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


def downsample_image_worker(input_path, output_path, target_size=(299, 299)):
    """
    Preprocess a single image to target size with high-quality downsampling.
    Default target size: (299, 299).

    Parameters:
    -----------
    input_path : str
        Path to the input image file.
    output_path : str
        Path where the downsampled image will be saved.
    target_size : tuple, optional
        Target size for downsampling, default is (299, 299).
    """

    try:
        # Read the image
        img = cv2.imread(input_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False

        # Resize with INTER_AREA for better quality when downsampling
        img_downsampled = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Save downsampled image
        cv2.imwrite(output_path, img_downsampled)
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def worker_wrapper(args):
    """
    Wrapper function that unpacks arguments for parallel image downsampling tasks.

    This function serves as an adapter between the ProcessPoolExecutor's map method
    and the downsample_image_worker function. It unpacks the tuple of arguments
    passed by the executor and forwards them to the worker function.

    Parameters:
    -----------
    args : tuple
        A tuple containing (input_path, output_path, target_size) where:
        - input_path : str
            Path to the source image file
        - output_path : str
            Path where the downsampled image will be saved
        - target_size : tuple
            Target dimensions (width, height) for the downsampled image

    Returns:
    --------
    bool
        True if downsampling was successful, False otherwise
    """
    input_path, output_path, target_size = args
    return downsample_image_worker(input_path, output_path, target_size)


def downsample_image_parallel(
    df, base_path="./images/image_train/", target_size=(299, 299), n_workers=None
):
    """
    Downsample product images contained in a DataFrame to target_size using parallel processing.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' in index and column for 'imageid'
    base_path : str, optional
        Directory where images are stored
        Default is './images/image_train/'
    target_size : tuple, optional
        Target size for downsampling
        Default is (299, 299)
    n_workers : int, optional
        Number of worker processes to use.
        If None, uses all available CPU cores.

    Returns:
    --------
    tuple
        (total_images, successful_images) counts
    """

    # Determine number of workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count()

    # Add suffix _ds to the base path as in "_downsampled"
    input_folder = base_path
    output_folder = re.sub(r"\/([^\/]*)$", r"_ds/\1", input_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a list of tasks for ProcessPoolExecutor
    worker_args = []

    # Process each row in the input DataFrame
    for idx, row in df.iterrows():
        # Get productid and imageid from the DataFrame
        # Construct the input and output paths based on your DataFrame structure
        # Adjust the filename construction based on your actual image naming convention
        product_id = idx
        image_id = row["imageid"]

        # Construct the file paths
        input_path = os.path.join(
            input_folder, f"image_{image_id}_product_{product_id}.jpg"
        )

        output_path = os.path.join(
            output_folder, f"image_{image_id}_product_{product_id}_ds.jpg"
        )

        # Only add if the input file exists
        if os.path.exists(input_path):
            worker_args.append((input_path, output_path, target_size))

    print(
        f"\033[1mdownsample_image_parallel()\033[0m: Found {len(worker_args):,} images to process."
    )

    # Downsample images in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process results with a progress bar
        results = list(
            tqdm(
                executor.map(worker_wrapper, worker_args),
                total=len(worker_args),
                desc="Downsampling images",
            )
        )

    # Count successful operations
    successful = sum(1 for result in results if result)

    print(f"Successfully downsampled {successful} of {len(worker_args)} images")
    return len(worker_args), successful


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

    # Save the updated DataFrame
    df.to_csv("df_image.csv", index=True)

    return df
