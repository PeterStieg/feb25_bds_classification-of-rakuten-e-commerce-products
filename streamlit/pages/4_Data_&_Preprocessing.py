# Data & Preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import unicodedata

# Add project root to path to import streamlit_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Data & Preprocessing",
    page_icon="streamlit/assets/images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(4 / 7)
st.title("Data Exploration & Preprocessing")
st.sidebar.header(":material/data_exploration: Data & Preprocessing")
st.sidebar.image("streamlit/assets/images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Data exploration tabs
overview_tab, hypotheses_tab, text_tab, image_tab = st.tabs(
    ["Dataset Overview", "Research Hypotheses", "Text Preprocessing", "Image Preprocessing"]
)

with overview_tab:
    st.markdown("""
    ## Dataset Characteristics
    
    ### Rakuten France Product Catalog
    - **Training samples**: 84,916 products
    - **Product categories**: 27 distinct product type codes
    - **Modalities**: Text (designation + description) + Product images
    - **Language diversity**: French, German, and international content
    - **Real-world complexity**: Missing descriptions, merchant-generated content
    """)
    
    # Dataset structure
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Data Structure
        
        **Text Features:**
        - `designation`: Product title (always present)
        - `description`: Detailed description (~35% missing)
        - `productid`: Unique product identifier
        - `imageid`: Associated image identifier
        
        **Target Variable:**
        - `prdtypecode`: Product classification (27 categories)
        """)
    
    with col2:
        st.markdown("""
        ###  Data Quality Assessment
        
        **Text Data:**
        - **Coverage**: 100% products have titles
        - **Missing descriptions**: ~35% of products
        - **Language distribution**: 76.6% French, 23.4% non-French
        - **Content quality**: Real-world merchant variability
        
        **Image Data:**
        - **Coverage**: 100% products have images
        - **Original quality**: 2-104 KB file sizes (mean: 26 KB)
        - **Post-processing**: 98.3% meet CNN training standards
        """)
    
    # Category distribution
    st.markdown("""
    ###  Product Category Distribution
    """)
    
    # Display category distribution charts
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        try:
            st.image("streamlit/assets/images/parent_category_distribution.png",
                    caption="Parent Category Distribution (8 major groups)",
                    width=400)
        except FileNotFoundError:
            st.info(" Parent category distribution chart will display here")
    
    with cat_col2:
        try:
            st.image("streamlit/assets/images/subcategory_distribution.png",
                    caption="Subcategory Distribution (All 27 categories)",
                    width=400)
        except FileNotFoundError:
            st.info(" Subcategory distribution chart will display here")
    
    st.markdown("""
    **Parent Categories (8 major groups):**
    - Books: 24.0% (20,381 products)
    - Toys & Children: 21.8% (18,491 products) 
    - Garden & Pool: 18.0% (15,312 products)
    - Video Games: 12.3% (10,450 products)
    - Sports & Travel: 9.4% (8,018 products)
    - Home & Lighting: 8.7% (7,408 products)
    - Pet Store: 1.0% (851 products)
    - Wines & Gastronomy: 0.9% (781 products)
    
    **Classification Challenge:**
    - **27 subcategories**with significant class imbalance (13.4:1 ratio)
    - **Moderate parent imbalance**: Largest category 25x larger than smallest
    - **Real-world e-commerce distribution**reflecting actual marketplace patterns
    """)

with hypotheses_tab:
    st.markdown("""
    ## Research Hypotheses (H1-H7)
    
    Based on systematic data exploration, we formed testable hypotheses about classification difficulty and model performance patterns.
    """)
    
    # Hypotheses in columns
    hyp_col1, hyp_col2 = st.columns(2)
    
    with hyp_col1:
        st.markdown("""
        ###  Category Structure Hypotheses
        
        **H1: Inter-parent classification is easier**
        - Books vs Video Games vs Toys should achieve high accuracy
        - Different domains have distinct vocabulary and visual characteristics
        - *Prediction: High accuracy for parent-level classification*
        
        **H2: Intra-parent classification is harder**
        - Subcategories within same parent share vocabulary/appearance
        - Fine-grained distinctions will be more challenging
        - *Prediction: Lower accuracy for subcategory-level classification*
        
        **H3: Image features help with fine-grained distinctions**
        - VGG16 should outperform text models for subcategory classification
        - Visual differences may be clearer than textual differences
        - *Prediction: VGG16 > Text models for similar subcategories*
        
        **H4: Rare single-subcategory parents are easier**
        - Categories like Wines & Pet Store have only 1 subcategory each
        - High internal similarity despite low sample counts
        - *Prediction: High precision for rare categories despite fewer samples*
        """)
    
    with hyp_col2:
        st.markdown("""
        ###  Performance Prediction Hypotheses
        
        **H5: Subcategory complexity affects performance**
        - Parent categories with more subcategories may be harder to classify
        - More internal diversity creates classification challenges
        - *Prediction: Negative correlation between num_subcategories and avg_f1_score*
        
        **H6: Large subcategories achieve better classification performance**
        - Pool & Spa Maintenance (12.0%) should achieve higher F1 scores than rare categories
        - Top 5 subcategories should outperform bottom 5 subcategories
        - More training examples provide better feature learning
        - *Prediction: Strong positive correlation between subcategory size and F1 score*
        
        **H7: Visual characteristics vs quantity drive performance**
        - Categories show varying visual homogeneity (e.g., books vs diverse products)
        - Classification performance correlation with class size indicates whether sample quantity or intrinsic visual characteristics drive model accuracy
        - *Prediction: If sample quantity doesn't predict performance, visual characteristics are more important*
        """)
    
    st.info("""
    **Hypothesis Validation Framework**
    
    These hypotheses will be systematically tested using the final model performance results:
    - **H1-H2**: Compare inter-parent vs intra-parent classification accuracy
    - **H3**: Compare VGG16 vs text model performance on fine-grained categories
    - **H4**: Analyze rare category performance vs sample size
    - **H5**: Correlation analysis between subcategory complexity and performance
    - **H6**: Statistical relationship between category size and F1-score
    - **H7**: VGG16-specific analysis of visual homogeneity vs sample quantity
    
    *Results and validation presented in Models & Results section...*
    """)

with text_tab:
    st.markdown("""
    ## Text Preprocessing Pipeline
    
    ###  Data Quality Analysis Results
    """)
    
    # Display text length analysis charts
    length_col1, length_col2, length_col3 = st.columns(3)
    
    with length_col1:
        try:
            st.image("streamlit/assets/images/title_length_distribution.png",
                    caption="Title Length Distribution - Bimodal Pattern",
                    width=300)
        except FileNotFoundError:
            st.info(" Title length distribution chart will display here")
    
    with length_col2:
        try:
            st.image("streamlit/assets/images/description_length_distribution.png",
                    caption="Description Length Distribution - Long Tail Issue",
                    width=300)
        except FileNotFoundError:
            st.info(" Description length distribution chart will display here")
    
    with length_col3:
        try:
            st.image("streamlit/assets/images/description_outliers_boxplot.png",
                    caption="Description Length Outliers - Quality Issues",
                    width=300)
        except FileNotFoundError:
            st.info(" Description outliers boxplot will display here")
    
    st.markdown("""
    **Title (Designation) Quality:**
    -  **Well-controlled**: Platform enforces 250-character limit
    -  **Clean distribution**: Bimodal pattern (40-50 and 80-90 character peaks)
    -  **Minimal issues**: No extreme outliers or data quality problems
    -  **100% coverage**: All products have titles
    
    **Description Quality Issues:**
    -  **Uncontrolled lengths**: Range from 0 to 12,451 characters
    -  **Quality issues**: HTML artifacts, pseudo-empty content (`<br />`), formatting problems
    -  **35% missing data**: Requires graceful handling strategy
    -  **Long tail (3.46%)**: HTML artifacts and unprocessed web formatting
    """)
    
    st.markdown("""
    ###  Three-Stage Processing Pipeline
    
    Our preprocessing strategy creates model-specific text optimized for each architecture:
    """)
    
    # Display text preprocessing pipeline diagram
    st.markdown("###  Text Preprocessing Pipeline Flow")
    
    try:
        st.image("streamlit/assets/images/Text_preprocessing.png", 
                caption="Text Preprocessing Pipeline - German Product Example",
                width=600)
    except FileNotFoundError:
        st.info(" Text preprocessing pipeline diagram will display here")
    except Exception as e:
        st.warning(f"Error loading preprocessing diagram: {e}")
    
    st.markdown("""
    **Pipeline Key Points:**
    - **Example shows German product**: Demonstrates the translation workflow for 23.4% of products
    - **Progressive processing**: Each step adds specific preprocessing for target models
    - **Model-specific outputs**: BERT preserves context, Classical ML emphasizes clean features
    - **Real product transformation**: Shows actual text changes through the pipeline
    """)
    
    # Create three columns for preprocessing stages
    proc_col1, proc_col2, proc_col3 = st.columns(3)
    
    with proc_col1:
        st.markdown("""
        **1 Raw Text**
        
        ```python
        # Base combination
        raw_text = designation + description
        
        # Issues present:
        - HTML artifacts
        - Formatting problems  
        - Mixed languages
        - Inconsistent casing
        ```
        
        **Characteristics:**
        - Unprocessed merchant content
        - Real-world data complexity
        - Requires cleaning for ML use
        """)
    
    with proc_col2:
        st.markdown("""
        **2 BERT-Ready Text**
        
        ```python
        # Minimal preprocessing
        - HTML artifact removal
        - Language detection
        - DeepL translation (23.4% non-French)
        - Basic text combination
        
        # Preserves:
        - Contextual information
        - Natural language structure
        - Accent marks (native French)
        ```
        
        **Optimization for CamemBERT:**
        - Preserves linguistic context
        - Native French accent handling
        - Minimal preprocessing maintains semantic richness
        """)
    
    with proc_col3:
        st.markdown("""
        **3 Classical ML Text**
        
        ```python
        # Comprehensive preprocessing
        - All BERT preprocessing +
        - Accent normalization (unidecode)
        - French stop words removal
        - Lowercasing (via TF-IDF)
        - Feature filtering (len > 2)
        
        # Result:
        - Clean feature vectors
        - Optimized for TF-IDF
        ```
        
        **Optimization for SVM:**
        - Statistical pattern focus
        - Noise reduction through filtering
        - Consistent feature representation
        """)
    
    st.markdown("""
    ###  Preprocessing Results
    
    **Cleaning Effectiveness:**
    - **Average length reduction**: 594.7 → 525.0 characters (11.7% efficiency gain)
    - **Total cleanup**: 5.9 million characters of HTML/formatting artifacts removed
    - **Outlier reduction**: 24% reduction in extreme description lengths
    - **Quality improvement**: HTML entities properly decoded, formatting standardized
    
    **Translation Integration:**
    - **Language coverage**: 89.0% translation coverage for non-French content
    - **Multilingual handling**: 23.4% of products translated from German/other languages
    - **Consistency**: Unified French language processing across all models
    
    **Model-Ready Output:**
    -  **100% coverage**: All 84,916 products have processable text
    -  **No data loss**: Missing descriptions handled by title-only processing
    -  **Optimized formats**: Different preprocessing for each model architecture
    """)
    
    # Visual validation section
    st.markdown("""
    ###  Visual Validation: Word Cloud Comparison
    
    Demonstrating the preprocessing pipeline effectiveness using Books category as example:
    """)
    
    # Display pre-generated word cloud image
    try:
        st.image("streamlit/assets/images/wordcloud_comparison.png", 
                caption="Text Processing Pipeline Comparison - Books Category",
                width=700)
    except FileNotFoundError:
        st.info("""
        **Word Cloud Visualization Missing**
        
        To add the word cloud comparison:
        1. Run your word cloud code in notebook 01 section 10
        2. Save the plot: `plt.savefig('streamlit/assets/images/wordcloud_comparison.png', dpi=300, bbox_inches='tight')`
        3. The image will display here automatically
        
        This shows the three preprocessing stages:
        - **Raw Text**: Unprocessed product data with formatting artifacts
        - **BERT-Ready Text**: Clean content optimized for transformers  
        - **Classical ML Text**: Content words emphasized through French stopword removal
        """)
    except Exception as e:
        st.warning(f"Error loading word cloud image: {e}")
        st.info("Please ensure wordcloud_comparison.png is in streamlit/assets/images/ folder")
    
    st.markdown("""
    **Key Visual Insights from Word Clouds:**
    - **Raw text**shows unprocessed product data with formatting artifacts and inconsistent casing
    - **BERT-ready text**displays cleaned content optimized for transformer models with minimal preprocessing
    - **Classical ML text**emphasizes content words through comprehensive preprocessing including French stopword removal and accent normalization
    
    **Model-Specific Optimization:**
    Each preprocessing approach is tailored to its target model architecture - BERT benefits from contextual information preservation, while classical ML models perform better with normalized, filtered text features.
    """)
    
    st.markdown("""
    **Key Visual Insights from Word Clouds:**
    - **Raw text**shows unprocessed product data with formatting artifacts and inconsistent casing
    - **BERT-ready text**displays cleaned content optimized for transformer models with minimal preprocessing
    - **Classical ML text**emphasizes content words through comprehensive preprocessing including French stopword removal and accent normalization
    
    **Model-Specific Optimization:**
    Each preprocessing approach is tailored to its target model architecture - BERT benefits from contextual information preservation, while classical ML models perform better with normalized, filtered text features.
    """)

with image_tab:
    st.markdown("""
    ## Image Preprocessing Pipeline
    
    ###  Image Data Quality Assessment
    
    **Complete Image Coverage & Data Integrity:**
    - **100% availability**: All 84,916 training products have corresponding images
    - **Perfect text-image alignment**: Text and image modalities maintain identical class distribution
    - **Original quality range**: 2-104 KB file sizes (mean: 26 KB, std: 13.5 KB) indicating good compressed quality
    - **Format consistency**: Standardized JPEG format across all product images
    - **No corrupted files**: Zero processing errors across entire dataset
    """)
    
    # Display file size distribution
    try:
        st.image("streamlit/assets/images/image_file_size_distribution.png",
                caption="Image File Size Distribution (2-104 KB range, mean: 26 KB)",
                width=600)
    except FileNotFoundError:
        st.info(" Image file size distribution will display here")
        st.info("Expected file: streamlit/assets/images/image_file_size_distribution.png")
    except Exception as e:
        st.warning(f"Error loading file size distribution: {e}")
        st.info("File path: streamlit/assets/images/image_file_size_distribution.png")
    
    st.markdown("""
    **Class Distribution Validation:**
    - Identical class imbalance patterns between text and image modalities
    - Top categories range from ~4K to ~10K images each
    - Perfect correspondence validates multimodal dataset integrity
    - Consistent class imbalance effects across both channels
    
    ###  Automated Processing Pipeline
    
    **Four-Stage Processing Architecture:**
    """)
    
    # Image processing steps
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("""
        **1 Quality Analysis & Metadata Extraction**
        - File properties and size analysis (2-104 KB range)
        - Quality metrics assessment and validation
        - Basic image characteristics analysis
        - Perceptual hashing for duplicate detection
        - 100% successful metadata extraction
        
        **2 Intelligent Product Focus Detection**
        - Automated bounding box detection using OpenCV
        - **90.7% effective object detection rate**(77,021 actual objects detected)
        - **43.1% pixel reduction**through background removal
        - Smart cropping to product focus areas
        - Average detected object size: 356×360 pixels
        """)
    
    with img_col2:
        st.markdown("""
        **3 CNN-Optimized Processing**
        - Two-stage resize: Initial processing to 299×299, final VGG16 resize to 224×224
        - **58.5% downscaled**(49,654 images from larger bounding boxes)
        - **41.3% upscaled**(35,044 images from smaller bounding boxes)
        - **1.7% excluded**(1,412 images below 75px quality threshold)
        - ImageNet normalization statistics applied
        
        **4 Training Organization & Quality Control**
        - PyTorch-compatible class folder structure creation
        - Stratified train/validation splits (80/20, random_state=42)
        - **67,921 training + 16,995 validation images**
        - **98.3% quality standard compliance**for CNN training
        """)
    
    st.markdown("""
    ###  Processing Performance & Results
    
    **Image Processing Pipeline Demonstration:**
    """)
    
    # Display image processing pipeline visualization
    try:
        st.image("streamlit/assets/images/image_processing_pipeline.png",
                caption="Image Processing Pipeline: Raw → Bounding Box → Cropped & Resized",
                width=800)
    except FileNotFoundError:
        st.info(" Image processing pipeline demonstration will display here")
        st.info("Expected file: streamlit/assets/images/image_processing_pipeline.png")
    except Exception as e:
        st.warning(f"Error loading processing pipeline: {e}")
        st.info("File path: streamlit/assets/images/image_processing_pipeline.png")
    
    st.markdown("""
    **High-Speed Parallel Processing:**
    - **Processing rate**: 546 images/second with parallel processing
    - **Total processing time**: 2 minutes 35 seconds for complete dataset
    - **Success rate**: 100% successful processing (84,916/84,916 images)
    - **Zero processing failures**across entire pipeline
    
    **Bounding Box Detection Analysis:**
    - **Detection accuracy**: 90.7% effective object detection vs. 9.3% full-image boxes
    - **Data compression**: Average 1.8:1 compression ratio (131,328 vs 250,000 pixels)
    - **Background removal**: 9.14 billion pixels eliminated while preserving product information
    - **Object characteristics**: Average aspect ratio 1.149 (slightly wider than tall)
    
    **Storage Optimization Achievements:**
    - **Efficiency improvement**: 60-70% storage reduction achieved
    - **Smart processing**: Eliminated redundant intermediate files automatically
    - **Final storage**: Reduced from ~15GB to ~4-7GB while maintaining full functionality
    - **Automated cleanup**: Temporary processing files removed after completion
    
    ###  VGG16-Ready Dataset Structure
    
    **Final Organization:**
    ```
    data/processed/images/image_train_vgg16/
    ├── train/              # 67,921 images across 27 class folders
    │   ├── class_0/        # Organized by encoded labels (0-26)
    │   ├── class_1/        # Stratified sampling maintains class balance
    │   └── ...
    └── val/                # 16,995 images for validation
        ├── class_0/        # Same class structure as training
        ├── class_1/        # 20% stratified split from each category
        └── ...
    ```
    
    **Production Integration Features:**
    -  **PyTorch DataLoader compatible**: Direct integration with training pipeline
    -  **Stratified splits**: Balanced representation across all 27 categories  
    -  **Preprocessing applied**: All images optimized for VGG16 architecture (224×224)
    -  **Quality assured**: Only high-quality images (>75px) included in training set
    -  **Consistent random state**: random_state=42 ensures reproducible splits
    
    ###  Key Insights & Validation
    
    **Sample Product Images by Category:**
    """)
    
    # Display sample images by category
    try:
        st.image("streamlit/assets/images/sample_images_by_category.png",
                caption="Representative Product Images from Top 6 Categories",
                width=700)
    except FileNotFoundError:
        st.info(" Sample product images by category will display here")
        st.info("Expected file: streamlit/assets/images/sample_images_by_category.png")
    except Exception as e:
        st.warning(f"Error loading sample images: {e}")
        st.info("File path: streamlit/assets/images/sample_images_by_category.png")
    
    st.markdown("""
    **Within-Category Visual Consistency Analysis:**
    """)
    
    # Display within-category consistency comparison
    try:
        st.image("streamlit/assets/images/diverse_vs_homogeneous_categories.png",
                caption="Within-Category Consistency: Diverse vs. Homogeneous Categories",
                width=700)
    except FileNotFoundError:
        st.info(" Category consistency comparison will display here")
        st.info("Expected file: streamlit/assets/images/diverse_vs_homogeneous_categories.png")
    except Exception as e:
        st.warning(f"Error loading category comparison: {e}")
        st.info("File path: streamlit/assets/images/diverse_vs_homogeneous_categories.png")
    
    st.markdown("""
    **Dataset Authenticity:**
    - Significant diversity in image quality reflects real-world multi-merchant e-commerce platform
    - Professional product photography mixed with casual merchant uploads
    - Quality variation represents authentic classification challenges
    
    **Visual Classification Potential:**
    - Product categories show clear visual distinctiveness across different object types
    - Within-category consistency varies: some categories (books) show high homogeneity, others more diverse
    - **New Hypothesis H7**: Visual characteristics vs. sample quantity as performance drivers
    
    **Processing Pipeline Validation:**
    - High object detection rate (90.7%) validates bounding box preprocessing effectiveness
    - Substantial background removal improves focus on relevant product features
    - Size standardization enables consistent CNN input while preserving aspect ratios
    - Quality filtering (1.7% exclusion) balances data retention with training effectiveness
    """)
    

# Pagination and footer
add_pagination_and_footer("4_Data_Preprocessing.py")