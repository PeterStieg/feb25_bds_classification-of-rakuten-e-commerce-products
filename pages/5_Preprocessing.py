# Preprocessing
import streamlit as st
import re

import numpy as np
import pandas as pd

from PIL import Image
from text_utils import text_cleaner

df_text_clean = pd.read_csv("df_text_clean_streamlit.csv", index_col="productid")
df_text_preprocessing = df_text_clean[["designation", "description"]]

df_image_train = pd.read_csv("df_image_train.csv", index_col="productid")
df_image_train_preprocessing = df_image_train[["imageid"]]

st.set_page_config(
    page_title="Preprocessing",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("Preprocessing")
st.sidebar.header("Preprocessing")
st.markdown(
    """
Showcasing of preprocessing steps needed for...
1. **strings** contained in columns `designation` and `description`
2. **images** available in the directories 'image_train/' and 'image_test/'
"""
)

tab_text, tab_image = st.tabs(["1. Text preprocessing", "2. Image preprocessing"])

with tab_text:

    # Show examples from DataFrame with text data
    st.header("DataFrame with examples")

    # Selection text preprocessing step
    text_preprocessing_option = st.selectbox(
        "Select preprocessing step for strings:",
        (
            "Original text data",
            "1. Remove HTML tags and attributes",
            "2. Remove URLs, e.g. Amazon, AWS servers, etc.",
            "3. Remove control characters and problematic whitespace, e.g. '\x00-'",
            "4. Remove multiple question marks and inverted question marks",
            "5. Remove parentheses and quotes, e.g. (), [], \{\}",
        ),
    )

    if text_preprocessing_option == "1. Remove HTML tags and attributes":

        mask_html = (df_text_clean["description_html_tag"] == 1) | (
            df_text_clean["description_html_entity"] == 1
        )

        df_text_preprocessing = df_text_clean[mask_html][
            [
                "description",
                "description_cleaned",
                "description_html_tag",
                "description_html_entity",
            ]
        ]

    elif text_preprocessing_option == "2. Remove URLs, e.g. Amazon, AWS servers, etc.":

        mask_url = (
            (df_text_clean["description_URL"] == 1)
            & (df_text_clean["description_html_tag"] == 0)
            & (df_text_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_clean[mask_url][
            ["description", "description_cleaned", "description_URL"]
        ]

    elif (
        text_preprocessing_option
        == "3. Remove control characters and problematic whitespace, e.g. '\x00-'"
    ):
        mask_control_chars = (
            (df_text_clean["description_control_chars"] == 1)
            & (df_text_clean["description_URL"] == 0)
            & (df_text_clean["description_html_tag"] == 0)
            & (df_text_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_clean[mask_control_chars][
            ["description", "description_cleaned", "description_control_chars"]
        ]

    elif (
        text_preprocessing_option
        == "4. Remove multiple question marks and inverted question marks"
    ):
        mask_question_marks = df_text_clean["description_question_marks"] == 1

        df_text_preprocessing = df_text_clean[mask_question_marks][
            [
                "description",
                "description_cleaned",
                "description_question_marks",
                "description_question_mark_char_count",
            ]
        ]

        df_text_preprocessing = df_text_preprocessing.sort_values(
            by="description_question_mark_char_count", ascending=False
        )

    elif (
        text_preprocessing_option
        == "5. Remove parentheses and quotes, e.g. (), [], \{\}"
    ):
        mask_dashes = (
            (df_text_clean["description_parentheses"] == 1)
            & (df_text_clean["description_URL"] == 0)
            & (df_text_clean["description_html_tag"] == 0)
            & (df_text_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_clean[mask_dashes][
            ["description", "description_cleaned", "description_parentheses"]
        ]

    st.dataframe(df_text_preprocessing.head())

    # Showcase function from text_utils.py
    st.header("Showcase text cleaning function")
    text = st.text_input(
        "Try text preprocessing:",
        "Example   phrase     containing       spaces, tags and accents: <p>Caf&eacute;</p>.",
    )
    st.write("The cleaned text is:")
    cleaned_text = text_cleaner(text)
    st.code(cleaned_text, language="None")

with tab_image:

    # Show examples from DataFrame with image data
    st.header("DataFrame with examples")

    # Text model section
    image_preprocessing_option = st.selectbox(
        "Select step:",
        (
            "Original image data",
            "1. Basic image data extraction",
            "2. Bounding box detection",
            "3. Crop, pad and resize",
            "(4. Duplicate search)",
        ),
    )

    if image_preprocessing_option == "1. Basic image data extraction":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "prdtypecode",
                "file_size_kb",
                "mean_r",
                "mean_g",
                "mean_b",
            ]
        ]
    elif image_preprocessing_option == "2. Bounding box detection":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "prdtypecode",
                "bb_x",
                "bb_y",
                "bb_w",
                "bb_h",
                "bb_ar",
                "file_size_kb",
                "mean_r",
                "mean_g",
                "mean_b",
            ]
        ]
    elif image_preprocessing_option == "3. Crop, pad and resize":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "prdtypecode",
                "bb_x",
                "bb_y",
                "bb_w",
                "bb_h",
                "bb_ar",
                "file_size_kb",
                "mean_r",
                "mean_g",
                "mean_b",
                "downscale",
                "upscale",
                "exclude",
            ]
        ]
    elif image_preprocessing_option == "(4. Duplicate search)":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "prdtypecode",
                "bb_x",
                "bb_y",
                "bb_w",
                "bb_h",
                "bb_ar",
                "file_size_kb",
                "mean_r",
                "mean_g",
                "mean_b",
                "downscale",
                "upscale",
                "exclude",
                "phash",
                "phash_duplicate",
            ]
        ]

    # df_text_filtered = df_text_clean[text_preprocessing_option]
    st.dataframe(df_image_train_preprocessing.head())

    # Showcase function from image_utils.py
    st.header("Showcase image preprocessing function")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = Image.open(uploaded_image)

        # Convert to numpy array for OpenCV processing
        image_array = np.array(image)

        # Preprocessing options
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original image**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Bounding box**")
            st.image(image, use_container_width=True)
        with col3:
            st.markdown("**Crop, pad and resize**")
            st.image(image, use_container_width=True)
