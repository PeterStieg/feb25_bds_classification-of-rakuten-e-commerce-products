import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

"""
Text utility functions for text cleaning and preprocessing.

This module provides functions for tasks such as:
- Removing special characters
- Normalizing text
- Text combination and deduplication

These functions are designed to prepare text data for machine learning
and natural language processing tasks.
"""


def clean_malformed_html_tags(text):
    """
    Removes malformed paragraph markers at the beginning and end of text.
    Avoids replacing legitimate words.
    """
    # Return early for very short texts to avoid false positives
    if len(text) < 4:
        return text

    # Store original text in case we need to revert
    original_text = text

    # Only clean specific patterns that are very likely to be malformed HTML

    # 1. Clean "p" tags at beginning and end (must have uppercase letter after opening p)
    # The word boundary \b ensures we don't match 'p' at beginning of words like 'print'
    cleaned_text = re.sub(r"^\bp([A-Z].*?)p\b$", r"\1", text)

    # 2. Clean "pp" tags at beginning and end
    cleaned_text = re.sub(r"^\bpp(.*?)pp\b$", r"\1", cleaned_text)

    # 3. Clean "pdiv" at beginning and "divp" at end
    cleaned_text = re.sub(r"^\bpdiv(.*?)divp\b$", r"\1", cleaned_text)

    # 3.1 Additional common tag combinations
    cleaned_text = re.sub(r"^\bph5(.*?)h5p\b$", r"\1", cleaned_text)
    cleaned_text = re.sub(r"^\bp(.*?)divp\b$", r"\1", cleaned_text)
    cleaned_text = re.sub(r"^\bpdiv(.*?)p\b$", r"\1", cleaned_text)

    # Clean up extra whitespace and return
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def text_cleaner(input_data):
    """
    Clean text in pandas DataFrame or individual strings automatically.
    Simplified version without feature flags.

    Args:
        input_data: Either a DataFrame or a string to clean

    Returns:
        Either a cleaned DataFrame or a cleaned string
    """
    # Check if input is a DataFrame
    if isinstance(input_data, pd.DataFrame):
        return _text_cleaner_df(input_data)

    # Check if input is a string-like object
    elif isinstance(input_data, (str, bytes)):
        return _text_cleaner_str(str(input_data))

    else:
        raise TypeError("Input must be either a pandas DataFrame or a string")


def _text_cleaner_df(df):
    """
    Clean text in pandas DataFrame by removing unnecessary characters and normalizing spacing.
    Simplified version that only creates cleaned columns without feature flags.

    Args:
        df (pandas.DataFrame): The DataFrame containing text to clean

    Returns:
        pandas.DataFrame: The original DataFrame with cleaned text columns added
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Clean designation and description columns
    df_clean["designation_cleaned"] = df_clean["designation"].apply(_text_cleaner_str)
    df_clean["description_cleaned"] = df_clean["description"].fillna("").apply(_text_cleaner_str)

    return df_clean


def _text_cleaner_str(text):
    """
    Clean a single text string by removing HTML, URLs, and normalizing whitespace.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    cleaned_text = str(text)

    # Remove HTML tags with space and HTML entities with character
    cleaned_text = BeautifulSoup(cleaned_text, "html.parser").get_text(separator=" ")

    # Clean malformed HTML tags
    cleaned_text = re.sub(r"^p([A-Z].*)p$", r"\1", cleaned_text)

    # Remove URLs
    cleaned_text = re.sub(r"https?://[^\s\"'<>()]+", " ", cleaned_text)

    # Remove control characters and problematic whitespace
    cleaned_text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\xa0\u2000-\u200F\u2028-\u202F\u205F\u2060-\u206F\uFEFF]",
        " ",
        cleaned_text,
    )

    # Error patterns that can be replaced
    cleaned_text = re.sub(r'\\"', " ", cleaned_text)
    cleaned_text = re.sub(r"\\ \'", " ' ", cleaned_text)
    cleaned_text = re.sub(r"\\\'", "'", cleaned_text)

    # Remove multiple question marks and inverted question marks
    cleaned_text = re.sub(r"(\?{2,}|\¿(?=[\s\.,;:!?]))", " ", cleaned_text)

    # Remove multiple dashes or hyphens
    cleaned_text = re.sub(r"[-]{2,}", "-", cleaned_text)

    # Fix separator patterns
    cleaned_text = re.sub(r"(\S+)\s*(?://{2,}|\\\\+)\s+(\S+)", r"\1 \2", cleaned_text)

    # Remove whitespace characters
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    # Remove parentheses and quotes
    cleaned_text = re.sub(r"[\(\)\[\]\{\}‹›«»]", " ", cleaned_text)

    # Remove leading/trailing spaces and multiple spaces
    cleaned_text = re.sub(r"^\s+|\s+$", "", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    return cleaned_text


def count_matching_words(str1, str2):
    """Count matching words between two strings and return percentage."""
    words1 = str1.lower().split()
    words2 = str2.lower().split()
    common_words = [word for word in words1 if word in words2]
    return round(len(common_words) / len(words2) * 100, 2) if words2 else 0


def clean_description(str1, str2, cutoff=95):
    """Remove highly similar descriptions based on word overlap."""
    if pd.notna(str2):
        if pd.notna(str1):
            if count_matching_words(str1, str2) > cutoff:
                return np.nan
    return str2


def remove_duplicate_description_information(df, col1="designation", col2="description", cutoff=95):
    """Remove duplicate information between designation and description columns."""
    df[col2] = df.apply(lambda row: clean_description(row[col1], row[col2], cutoff=cutoff), axis=1)
    df[col1] = df.apply(lambda row: clean_description(row[col2], row[col1], cutoff=cutoff), axis=1)
    return df


def text_pre_processing(df):
    """
    Main text preprocessing function for model training.
    Simplified version without feature flags.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'designation' and 'description' columns
        
    Returns:
        pandas.DataFrame: Processed DataFrame with 'merged_text' column
    """
    df_processed = df.copy()
    
    # Store information if description is provided
    df_processed["bool_description"] = df_processed["description"].notnull().astype(int)
    
    # Clean text columns
    df_processed = text_cleaner(df_processed)
    
    # Use cleaned versions
    if "description_cleaned" in df_processed.columns:
        df_processed["description"] = df_processed["description_cleaned"]
    if "designation_cleaned" in df_processed.columns:
        df_processed["designation"] = df_processed["designation_cleaned"]
    
    # Remove duplicate descriptions
    df_processed = remove_duplicate_description_information(df_processed)
    
    # Create merged text
    df_processed["merged_text"] = np.where(
        df_processed["designation"].notna() 
        & df_processed["description"].notna() 
        & (df_processed["description"] != ""),
        df_processed["designation"] + " - " + df_processed["description"],
        df_processed["designation"].fillna("") + df_processed["description"].fillna(""),
    )
    
    return df_processed