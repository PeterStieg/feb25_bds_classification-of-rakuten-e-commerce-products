import os
import io
import requests
import tempfile
import json

import streamlit as st
import pandas as pd

from utils.image_utils import display_phash

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

# Constants
BUCKET_NAME = "feb25_bds_classification-of-rakuten-e-commerce-products"
GOOGLE_CLOUD_STORAGE_URL = "https://storage.googleapis.com"
GCP_PROJECT_URL = f"{GOOGLE_CLOUD_STORAGE_URL}/{BUCKET_NAME}"

PAGE_SEQUENCE = [
    {"name": "Home", "path": "Home.py"},
    {"name": "1. Team Presentation", "path": "pages/1_Team_Presentation.py"},
    {"name": "2. Project Outline", "path": "pages/2_Project_Outline.py"},
    {"name": "3. Modelling", "path": "pages/3_Modelling.py"},
    {"name": "4. Data Overview", "path": "pages/4_Data_Overview.py"},
    {"name": "5. Data Preprocessing", "path": "pages/5_Data_Preprocessing.py"},
    {"name": "6. Prediction", "path": "pages/6_Prediction.py"},
    {"name": "7. Thank you", "path": "pages/7_Thank_you.py"},
]

# List of product type codes necessary for function load_category_mapping
INDEX_TO_PRDTYPECODE = [
    "10",
    "2280",
    "2705",
    "2522",
    "2403",
    "50",
    "1140",
    "1180",
    "2462",
    "1160",
    "40",
    "60",
    "2905",
    "1280",
    "1300",
    "1320",
    "1302",
    "1301",
    "1281",
    "2582",
    "2583",
    "2585",
    "1560",
    "1920",
    "2060",
    "1940",
    "2220",
]


@st.cache_data
def load_DataFrame(URL):
    """
    Fetch a DataFrame from a given URL.

    Parameters:
    - URL: The URL to fetch the DataFrame from.

    Returns:
    - DataFrame: The fetched DataFrame.
    """
    df = pd.read_parquet(URL, engine="pyarrow")
    return df


@st.cache_resource
def load_vgg16():
    """
    Load a pretrained VGG16 model modified for FEB25 BDS Rakuten dataset classification.

    Downloads weights from GCP if not available locally and returns the model.
    """
    # Determine best available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start with pretrained VGG16 as model architecture
    model = models.vgg16(pretrained=True)

    # Modify classifier part, as model.state_dict
    # ... only contains the parameter values (weights and biases)
    # ... but not the architectural definitions
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(
        num_ftrs, 27
    )  # 27 outputs for categories of Rakuten dataset

    # Place the model on the best available device
    model = model.to(device)

    # Create a local path for downloading model weights
    vgg16_url = f"{GCP_PROJECT_URL}/model/vgg16_transfer_model.pth"
    local_path = os.path.join(tempfile.gettempdir(), "vgg16_transfer_model.pth")

    # Download file with model weights if it doesn't exist
    if not os.path.exists(local_path):

        try:
            response = requests.get(vgg16_url)
            response.raise_for_status()  # Check if download was successful

            # Save the content to the local file
            with open(local_path, "wb") as f:
                f.write(response.content)

        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None

    # Load checkpoint from local path
    try:
        checkpoint = torch.load(local_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Put the model in evaluation mode
        model.eval()
        return model

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def predict_vgg16(vgg16, image_path, radio_pred_class):
    """
    Predict the category of a product image using a pretrained VGG16 model.

    Parameters:
    - model: The pretrained VGG16 model.
    - image: The input image to classify.

    Returns:
    - category: The predicted category.
    """

    # Determine if image_path is a URL or local path
    # Download the image from URL
    if isinstance(image_path, str) and image_path.startswith("http"):

        try:
            response = requests.get(image_path)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        except Exception as e:
            st.error(f"Failed to download image: {str(e)}")
            return "Error loading image"

    # Open local image file
    elif isinstance(image_path, str):
        image = Image.open(image_path)

    # Assume it's already a PIL Image or similar
    else:
        image = image_path

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image)

    input_batch = input_tensor.unsqueeze(
        0
    )  # Create a mini-batch as expected by the model

    # Get the device from one of the model parameters instead of directly from the model
    device = next(vgg16.parameters()).device

    # Move the input to the same device as the model
    input_batch = input_batch.to(device)

    # Perform inference
    with torch.no_grad():
        output = vgg16(input_batch)

    # Get the predicted category
    _, predicted_idx = torch.max(output, 1)
    index = predicted_idx.item()

    # Convert the index to a product type code
    prdtypecode = INDEX_TO_PRDTYPECODE[index]

    # Convert the product type code to a category name
    category_mapping = load_category_mapping(language="en")
    category_name = category_mapping.get(
        prdtypecode, f"Unknown Category ({prdtypecode})"
    )

    if radio_pred_class == "Primary Category":
        return category_name.split(" > ")[0]
    elif radio_pred_class == "Subcategory":
        return category_name.split(" > ")[-1]


@st.cache_data
def load_category_mapping(language="en"):
    """
    Load the mapping from product type codes to category names.

    Parameters:
    - language: The language for the category names (default: "en").

    Returns:
    - dict: The mapping from product type codes to category names.
    """
    with open("data/prdtypecode_to_category_name.json", "r") as f:
        mapping = json.load(f)
    return mapping[language]


def add_pagination_and_footer(current_page_path):
    # Find current page index in sequence
    current_index = next(
        (
            i
            for i, page in enumerate(PAGE_SEQUENCE)
            if page["path"] == current_page_path
        ),
        0,
    )

    # Horizontal line
    st.divider()

    # Create columns for previous, current page indicator, next
    prev_butt, next_butt = st.columns(
        2
    )  # elf* and erc* as in "empty left column" and "empty right column"

    # Previous button
    with prev_butt:
        if current_index > 0:  # Not on first page
            prev_page = PAGE_SEQUENCE[current_index - 1]
            if st.button("← Previous", use_container_width=True):
                st.switch_page(prev_page["path"])

    # Next button
    with next_butt:
        if current_index < len(PAGE_SEQUENCE) - 1:  # Not on last page
            next_page = PAGE_SEQUENCE[current_index + 1]
            if st.button("Next →", use_container_width=True):
                st.switch_page(next_page["path"])

    # Copyright line and page indicator
    st.markdown(
        f"© 2025 // Peter Stieg, Robert Wilson, Thomas Borer // [Page {current_index + 1}/{len(PAGE_SEQUENCE)}]"
    )


def display_image(id, option):  # id as in "image data"

    if option == "preprocessing":

        dataset = "train"
        pi = id[0]  # product ID
        ii = id[1]  # image ID
        pt = id[2]  # product title, aka designation
        ipo = id[3]  # image preprocessing option
        iph = id[4]  # image perceptual hash
        fais = ""

        if ipo == "2. Bounding box detection":
            fais = "_bb"
        elif ipo == "3. Crop, pad and resize":
            fais = "_cpr"
        else:
            pass

        # Create the full public URL for the image
        if ipo != "(4. Duplicate search)":
            image_path = f"image_{ii}_product_{pi}{fais}.jpg"
            image_url = f"{GCP_PROJECT_URL}/images/image_{dataset}{fais}/{image_path}"
        else:
            image_url = display_phash(iph, size=8, scale=32)

        # Display the image with product ID and image ID as caption
        st.image(
            image_url,
            caption=f"Title: {pt[:33]} - Product: {pi} - Image: {ii}",
            use_container_width=True,
        )

    # Test data: Display the processed image without caption
    elif option == "prediction_proc":

        dataset = "test"
        pi = id[0]  # product ID
        ii = id[1]  # image ID
        pt = id[2]  # product title, aka designation
        fais = "_cpr"

        image_path = f"image_{ii}_product_{pi}{fais}.jpg"
        image_url = f"{GCP_PROJECT_URL}/images/image_{dataset}{fais}/{image_path}"

        st.image(
            image_url,
            use_container_width=True,
        )

    # Test data: Display the processed image without caption
    elif option == "prediction_org":

        dataset = "test"
        pi = id[0]  # product ID
        ii = id[1]  # image ID
        pt = id[2]  # product title, aka designation
        fais = ""

        image_path = f"image_{ii}_product_{pi}{fais}.jpg"
        image_url = f"{GCP_PROJECT_URL}/images/image_{dataset}{fais}/{image_path}"

        st.image(
            image_url,
            use_container_width=True,
        )


def get_img_path(pi, ii, option):
    """
    Generate the image path for a cropped, padded and resized image.

    Parameters:
    - pi: Product ID
    - ii: Image ID

    Returns:
    - str: The image path.
    """
    if option == "cpr":
        return f"{GCP_PROJECT_URL}/images/image_test_{option}/image_{ii}_product_{pi}_{option}.jpg"
