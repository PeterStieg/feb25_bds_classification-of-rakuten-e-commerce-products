import re

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning

# Supress warnings from BeautifulSoup parser as we are not using it for parsing HTML
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

"""
Text utility functions for text cleaning and preprocessing.

This module provides functions for tasks such as:
- Removing special characters

Planned features:
- Normalizing text
- Tokenization helpers
- Stopword removal
- Lemmatization and stemming utilities

These functions are designed to prepare text data for machine learning
and natural language processing tasks.
"""


def clean_malformed_html_tags(text):
    """
    Removes leading and trailing malformed HTML tags.
    Handles tag combinations and internal paragraph markers.
    """
    # Define common HTML tags to look for
    html_tags = [
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "span",
        "br",
        "ul",
        "li",
        "a",
        "img",
        "table",
        "tr",
        "td",
        "th",
    ]

    # Generate patterns for tag combinations (up to 2 tags combined)
    tag_combinations = []
    for tag1 in html_tags:
        tag_combinations.append(tag1)
        for tag2 in html_tags:
            tag_combinations.append(tag1 + tag2)

    # Create regex pattern for tags and combinations
    tag_pattern = "|".join(tag_combinations)

    # Step 1: Remove leading tag combinations
    # Match one or more occurrences of tag-like strings at the beginning
    leading_pattern = r"^((?:" + tag_pattern + r"){1,2})"
    cleaned_text = re.sub(leading_pattern, "", text)

    # Step 2: Remove trailing tag combinations
    # Match one or more occurrences of tag-like strings at the end
    trailing_pattern = r"((?:" + tag_pattern + r"){1,2})$"
    cleaned_text = re.sub(trailing_pattern, "", cleaned_text)

    # Step 3: Clean up internal paragraph markers
    # This preserves paragraph structure while removing tags
    internal_p_pattern = r"\s*p\s*p\s*"
    cleaned_text = re.sub(internal_p_pattern, "\n\n", cleaned_text)

    # Step 4: Handle combined tags within the text
    # Replace tag combinations with appropriate spacing
    for combo in tag_combinations:
        if len(combo) > 1:  # Only process combinations, not single tags
            pattern = r"\s*" + re.escape(combo) + r"\s*"
            if "p" in combo:  # If it contains 'p', treat it as a paragraph break
                cleaned_text = re.sub(pattern, "\n\n", cleaned_text)
            else:  # Otherwise just replace with a space
                cleaned_text = re.sub(pattern, " ", cleaned_text)

    # Clean up extra whitespace and return
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def text_cleaner(df):
    """
    Clean text in pandas DataFrame by removing unnecessary characters and normalizing spacing.

    Cleans 'designation' and 'description' columns and stores results in
    'designation_cleaned' and 'description_cleaned' columns. Also adds boolean flag columns
    to track text characteristics.

    Removes:
    - Leading and trailing spaces
    - HTML tags, e.g. \<br>, \<br />, \<b>

    Replaces:
    - Multiple spaces with a single space
    - HTML entities with their corresponding characters, e.g. &eacute; → è, &auml; → ä, &ntilde; → ñ
    - Control characters with empty strings, e.g. Ã, Â©, �
    - Non-ASCII characters with empty strings

    Args:
        df (pandas.DataFrame): The DataFrame containing text to clean

    Returns:
            pandas.DataFrame: The original DataFrame with additional columns:
                - designation_cleaned, description_cleaned: Cleaned text columns
                - designation_org_clean, description_org_clean: Overall cleanliness flags
                - Additional feature flags for HTML tags, HTML flags, spaces, uppercase, and lowercase text
    """

    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Create columns for cleaned text
    df_clean["designation_cleaned"] = ""
    df_clean["description_cleaned"] = ""

    # Create columns for string length
    designation_len = df_clean["designation"].str.len()
    description_len = df_clean["description"].str.len()

    df_clean["designation_len"] = designation_len
    df_clean["description_len"] = description_len
    df_clean["designation_cleaned_len"] = 0
    df_clean["description_cleaned_len"] = 0

    df_clean["designation_saved_len"] = 0
    df_clean["description_saved_len"] = 0

    # Create columns for word count
    df_clean["designation_word_count"] = df_clean["designation"].str.split().str.len()
    df_clean["description_word_count"] = df_clean["description"].str.split().str.len()
    df_clean["designation_cleaned_word_count"] = 0
    df_clean["description_cleaned_word_count"] = 0

    df_clean["designation_saved_words"] = 0
    df_clean["description_saved_words"] = 0

    # Create flag columns
    for col in ["designation", "description"]:

        df_clean[f"{col}_org_clean"] = 1  # Assume clean until proven dirty

        df_clean[f"{col}_spaces"] = 0
        df_clean[f"{col}_upper"] = 0
        df_clean[f"{col}_lower"] = 0

        df_clean[f"{col}_html_tag"] = 0
        df_clean[f"{col}_html_entity"] = 0

        df_clean[f"{col}_encoding_issue"] = 0
        df_clean[f"{col}_control_chars"] = 0

        df_clean[f"{col}_URL"] = 0
        df_clean[f"{col}_error_pattern"] = 0
        df_clean[f"{col}_separator"] = 0

        df_clean[f"{col}_parentheses"] = 0
        df_clean[f"{col}_question_marks"] = 0
        df_clean[f"{col}_question_mark_char_count"] = 0
        df_clean[f"{col}_hyphens"] = 0

    # Map original columns to their cleaned counterparts
    col_mapping = {
        "designation": "designation_cleaned",
        "description": "description_cleaned",
    }

    # Process each row
    for idx, row in df_clean.iterrows():
        for orig_col, clean_col in col_mapping.items():
            if orig_col not in row or pd.isna(row[orig_col]):
                continue

            text = str(row[orig_col])

            # Check uppercase
            if text.isupper():
                df_clean.at[idx, f"{orig_col}_upper"] = 1

            # Check lowercase
            if text.islower():
                df_clean.at[idx, f"{orig_col}_lower"] = 1

            # Check HTML tags
            if re.search(r"<[^>]+>", text):
                df_clean.at[idx, f"{orig_col}_html_tag"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check HTML entities
            if re.search(r"&[a-zA-Z0-9#]+;", text):
                df_clean.at[idx, f"{orig_col}_html_entity"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for potential encoding issues
            if re.search(r"Ã.|\xef\xbf\xbd|�", text):
                df_clean.at[idx, f"{orig_col}_encoding_issue"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for control characters and problematic whitespace
            if re.search(
                r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\xa0\u2000-\u200F\u2028-\u202F\u205F\u2060-\u206F\uFEFF]",
                text,
            ):
                df_clean.at[idx, f"{orig_col}_control_chars"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check extra spaces
            if re.search(r"^\s{1,}|\s{1,}$|\s{2,}", text):
                df_clean.at[idx, f"{orig_col}_spaces"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for URLs (both http and www)
            if re.search(r"https?://[^\s\"'<>()]+|www\.[^\s\"'<>()]+", text):
                df_clean.at[idx, f"{orig_col}_URL"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for error patterns: \"", \ ', \', ??
            if re.search(r'\\""|\\ \'|\\\'|\?{2,}', text):
                df_clean.at[idx, f"{orig_col}_error_pattern"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for separators like //, \\, or ////
            if re.search(r"\s+(?://{2,}|\\{2,})\s+", text):
                df_clean.at[idx, f"{orig_col}_separator"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for parentheses like (), [], {} and quotes like ‹› or « »
            if re.search(r"[\(\)\[\]\{\}‹›«»]", text):
                df_clean.at[idx, f"{orig_col}_parentheses"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Check for multiple normal question marks and inverted question marks like ?? and ¿
            if re.search(r"(\?{2,}|\¿(?=[\s\.,;:!?]))", text):
                df_clean.at[idx, f"{orig_col}_question_marks"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

                # Count question marks (both normal and inverted)
                question_mark_count = len(re.findall(r"[\?\¿]", text))
                df_clean.at[idx, f"{orig_col}_question_mark_char_count"] = (
                    question_mark_count
                )

            # Check for multiple dashes or hyphens
            if re.search(r"[-]{2,}", text):
                df_clean.at[idx, f"{orig_col}_hyphens"] = 1
                df_clean.at[idx, f"{orig_col}_org_clean"] = 0

            # Clean text
            cleaned_text = text

            # Remove HTML tags with space and HTML entities with character
            cleaned_text = BeautifulSoup(cleaned_text, "html.parser").get_text(
                separator=" "
            )

            # Clean malformed HTML tags, e.g. leading and trailing tags without anchor brackets
            # cleaned_text = clean_malformed_html_tags(cleaned_text)
            '''We cannot discover most malformed html_tags the only thing we can try to find is malformed paragraph markers if they occur at the beginning and end of the description '''
            cleaned_text = re.sub(r"^p([A-Z].*)p$", r"\1", cleaned_text)

            # Remove URLs
            cleaned_text = re.sub(r"https?://[^\s\"'<>()]+", " ", cleaned_text)

            # Remove control characters and problematic whitespace
            cleaned_text = re.sub(
                r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\xa0\u2000-\u200F\u2028-\u202F\u205F\u2060-\u206F\uFEFF]",
                " ",
                cleaned_text,
            )

            # Error patterns that can be replaced: Escaped quotes and apostrophes
            cleaned_text = re.sub(r'\\"', " ", cleaned_text)
            cleaned_text = re.sub(r"\\ \'", " ' ", cleaned_text)
            cleaned_text = re.sub(r"\\\'", "'", cleaned_text)

            # Remove multiple question marks and inverted question marks
            cleaned_text = re.sub(r"(\?{2,}|\¿(?=[\s\.,;:!?]))", " ", cleaned_text)

            # Remove multiple dashes or hyphens
            cleaned_text = re.sub(r"[-]{2,}", "-", cleaned_text)

            # Fix separator patterns
            cleaned_text = re.sub(
                r"(\S+)\s*(?://{2,}|\\\\+)\s+(\S+)", r"\1 \2", cleaned_text
            )

            # Remove whitespace characters e.g. \n, \r, \t
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)

            # Remove parentheses and quotes
            cleaned_text = re.sub(r"[\(\)\[\]\{\}‹›«»]", " ", cleaned_text)

            # Remove leading/trailing spaces
            cleaned_text = re.sub(r"^\s+|\s+$", "", cleaned_text)

            # Replace multiple spaces with a single space
            cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

            # Update the cleaned_text text column
            df_clean.at[idx, clean_col] = cleaned_text

    # Update columns for string length
    designation_cleaned_len = df_clean["designation_cleaned"].str.len()
    description_cleaned_len = df_clean["description_cleaned"].str.len()

    df_clean["designation_cleaned_len"] = designation_cleaned_len
    df_clean["description_cleaned_len"] = description_cleaned_len

    df_clean["designation_saved_len"] = designation_len - designation_cleaned_len
    df_clean["description_saved_len"] = description_len - description_cleaned_len

    # Create columns for word count
    df_clean["designation_cleaned_word_count"] = (
        df_clean["designation_cleaned"].str.split().str.len()
    )
    df_clean["description_cleaned_word_count"] = (
        df_clean["description_cleaned"].str.split().str.len()
    )

    # Return cleaned DataFrame
    return df_clean


def text_merger(df):
    """
    Merge text columns and set flags for empty or identical columns.

    This function:
    1. Checks if 'designation' or 'description' are empty and sets flags
    2. Checks if 'designation' or 'description' are identical and sets a flag
    3. Checks if cleaned versions 'designation_cleaned' or 'description_cleaned' are identical and sets a flag
    4. Creates a merged 'text_merged' column combining designation and description

    Args:
        df (pandas.DataFrame): DataFrame containing designation and description columns

    Returns:
        pandas.DataFrame: Original DataFrame with additional columns:
            - text_merged: Merged text from designation and description
            - designation_empty: Flag indicating if designation is empty
            - description_empty: Flag indicating if description is empty
            - identical_original: Flag indicating if original columns are identical
            - identical_cleaned: Flag indicating if cleaned columns are identical
    """

    # Create a copy to avoid modifying the original DataFrame
    df_merge = df.copy()

    # Create flag columns
    df_merge["designation_empty"] = 0
    df_merge["description_empty"] = 0
    df_merge["identical_original"] = 0
    df_merge["identical_cleaned"] = 0

    # Create merged text column
    df_merge["text_merged"] = ""

    # Process each row
    for idx, row in df_merge.iterrows():
        # Check for empty columns
        if "designation_cleaned" not in row or pd.isna(row["designation_cleaned"]):
            df_merge.at[idx, "designation_empty"] = 1

        if "description_cleaned" not in row or pd.isna(row["description_cleaned"]):
            df_merge.at[idx, "description_empty"] = 1

        # Get cleaned designation and cleaned description and handle missing values
        designation = (
            str(row["designation_cleaned"])
            if "designation_cleaned" in row and not pd.isna(row["designation_cleaned"])
            else ""
        )
        description = (
            str(row["description_cleaned"])
            if "description_cleaned" in row and not pd.isna(row["description_cleaned"])
            else ""
        )

        # Check if original columns are identical – if both are non-empty
        if designation and description and designation == description:
            df_merge.at[idx, "identical_original"] = 1

        # Check if cleaned columns are identical – if both exist
        if (
            "designation_cleaned" in row
            and "description_cleaned" in row
            and not pd.isna(row["designation_cleaned"])
            and not pd.isna(row["description_cleaned"])
        ):
            if row["designation_cleaned"] == row["description_cleaned"]:
                df_merge.at[idx, "identical_cleaned"] = 1

        # Create merged text
        if df_merge.at[idx, "description_empty"]:
            df_merge.at[idx, "text_merged"] = designation
        elif df_merge.at[idx, "identical_original"]:
            df_merge.at[idx, "text_merged"] = designation
        else:
            df_merge.at[idx, "text_merged"] = designation + " // " + description

    return df_merge


def count_matching_words(str1, str2):
    """count_matching_words(string1, string2) will return the percentage rounded to 2 decimal places"""
    words1 = str1.lower().split()
    words2 = str2.lower().split()
    common_words = [word for word in words1 if word in words2]  # Find matching words
    return (
        round(len(common_words) / len(words2) * 100, 2) if words2 else 0
    )  # Avoid division by zero


def clean_description(str1, str2, cutoff=95):
    if pd.notna(str2):  # Check if str2 is not NaN
        if pd.notna(str1):
            if count_matching_words(str1, str2) > cutoff:
                return np.nan  # Remove highly similar descriptions
    return str2  # Keep the original if not highly similar


def remove_duplicate_description_information(
    df, col1="designation", col2="description", cutoff=95
):
    """removes information from description and designation column if the information in description is already in designation or if there is more information in description than designation"""
    df[col2] = df.apply(
        lambda row: clean_description(row[col1], row[col2], cutoff=cutoff), axis=1
    )
    df[col1] = df.apply(
        lambda row: clean_description(row[col2], row[col1], cutoff=cutoff), axis=1
    )
    return df


def text_pre_processing(df):
    # store information if description is provided
    df["bool_description"] = df["description"].notnull().astype(int)
    # call text_cleaner function
    df = text_cleaner(df)
    # will continue to use description and designation columns as default
    if "description_cleaned" in df.columns:
        df.description = df.description_cleaned
    if "designation_cleaned" in df.columns:
        df.designation = df.designation_cleaned
    # remove 'duplicated' descriptions or designations
    df = remove_duplicate_description_information(df)
    # merge designation and description to merged_text, avoiding leading or trailing ' - ' while using it as seperators
    df["merged_text"] = np.where(
        df["designation"].notna() & df["description"].notna() & (df["description"] != ""),
        df["designation"] + " - " + df["description"],
        df["designation"].fillna("") + df["description"].fillna(""),
    )
    # clean up, remove designation and description columns
    df.drop(["designation", "description"], axis=1, inplace=True)
    return df


def hw():
    """Test function to print "Hello, world!".

    This function serves as a placeholder to demonstrate the module's structure.
    """
    print("Hello, world!")
