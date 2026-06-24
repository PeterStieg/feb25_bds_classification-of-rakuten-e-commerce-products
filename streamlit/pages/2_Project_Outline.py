# Project Outline
import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path to import streamlit_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Project Outline",
    page_icon="streamlit/assets/images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(2 / 7)
st.title("Project Outline")
st.sidebar.header(":material/work: Project Outline")
st.sidebar.image("streamlit/assets/images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Project overview tabs
overview_tab, challenge_tab, approach_tab, metrics_tab = st.tabs(
    ["Business Context", "Technical Challenge", "Our Approach", "Success Metrics"]
)

with overview_tab:
    st.markdown("""
    ## The Business Problem
    
    In today's fast-paced e-commerce landscape, the ability to accurately and efficiently classify products is critical for enhancing customer experience, streamlining operations, and driving revenue growth. E-commerce platforms like Rakuten, with over 1.3 billion users worldwide, face the fundamental challenge of product categorization at massive scale. The complexity arises from diverse product data, potential inconsistencies in merchant-generated text, and visual variability of product images across international marketplaces.
    
    Our solution approach combines business insights with cutting-edge technical methodologies to address this challenge comprehensively. From a business perspective, we aim to improve key processes such as product discoverability, inventory management, and personalized recommendations, all of which directly impact customer satisfaction and operational efficiency.
    
    ### Business Impact Areas
    
    **Enhanced Customer Experience**
    - **Product discoverability**: Customers find relevant products faster through improved search
    - **Search optimization**: Better internal search functionality and SEO performance  
    - **Personalized recommendations**: Accurate categories enable targeted product suggestions
    
    **Operational Excellence**
    - **Automated categorization**: Reduces manual effort and eliminates classification errors
    - **Faster product onboarding**: Accelerated time-to-market for new merchant listings
    - **Scalable operations**: Handles catalog growth without proportional staffing increases
    
    **Financial Performance**
    - **Revenue growth**: Improved discoverability drives higher conversion rates and sales
    - **Cost reduction**: Automation significantly reduces operational overhead and manual processing
    - **Inventory optimization**: Enhanced demand forecasting and stock management capabilities
    - **Marketing effectiveness**: Enables targeted campaigns with measurably higher ROI
    
    ### The Scalability Challenge
    
    Manual and rule-based categorization approaches fail to meet modern e-commerce demands:
    - **Mixed merchant ecosystem**with professional and non-professional sellers
    - **Multilingual content**spanning French, German, and international languages
    - **Product diversity**across hundreds of distinct categories and subcategories
    - **Real-world data complexity**including missing descriptions and inconsistent labeling
    
    ### Rakuten France Classification Challenge
    
    The specific technical challenge involves categorizing products into **27 distinct product type codes**using:
    - **Product titles**(designation) - merchant-provided short descriptions
    - **Product descriptions**- detailed information (missing in ~35% of cases)
    - **Product images**- visual representation requiring computer vision analysis
    
    **Example**: *"Klarstein Présentoir 2 Montres Optique Fibre"* + product image → Category 1500
    
    **Strategic Objective**: Develop an automated, scalable classification solution that improves accuracy while substantially reducing manual categorization overhead and operational costs.
    """)

with challenge_tab:
    st.markdown("""
    ## Technical Challenge Details
    
    ### Dataset Characteristics
    - **84,916 training products**from Rakuten France catalog
    - **27 product categories**(prdtypecode classification)
    - **Multimodal data**: Text (French/German) + Product images
    - **Real-world complexity**: Missing descriptions (~35%), noisy labels, unbalanced distribution
    
    ### Data Structure
    ```
    X_train.csv: Product information
    ├── designation: Product title
    ├── description: Detailed description (often NaN)
    ├── productid: Unique product identifier  
    └── imageid: Associated image identifier
    
    Y_train.csv: Target categories
    └── prdtypecode: Product type classification (27 categories)
    
    Images: 84,916 product images
    └── Format: image_[imageid]_product_[productid].jpg
    ```
    
    ### Research Challenges
    
    **Multimodal Complexity**
    - How to effectively combine text and visual information?
    - Different modalities may provide conflicting signals
    
    **Data Quality Issues**
    - Missing descriptions in ~35% of products
    - Noisy, real-world merchant-generated content
    - Multilingual text requiring translation/normalization
    
    **Class Imbalance**
    - Uneven distribution across 27 categories
    - Some categories with thousands of samples, others with few
    
    **Official Benchmarks to Compare Against**
    - **Text CNN**: 0.8113 F1-score (weighted)
    - **Image ResNet50**: 0.5534 F1-score (weighted)
    """)

with approach_tab:
    st.markdown("""
    ## Our Solution Strategy
    
    ### Core Hypothesis
    **Different model architectures capture complementary aspects of product information**that can be systematically combined for superior classification performance.
    
    ### High-Level Approach: Multimodal Ensemble
    
    We developed a **three-model ensemble system**that combines:
    
    **Classical Machine Learning**
    - Robust statistical pattern recognition in text
    - TF-IDF vectorization with optimized algorithms
    
    **Deep Learning (BERT)**
    - Contextual understanding of French product descriptions
    - Pre-trained transformer fine-tuned for classification
    
    **Computer Vision**
    - Visual product characteristics analysis
    - Transfer learning from pre-trained CNN models
    
    ### Why This Approach Works
    
    **Complementary Strengths**: Each model type excels at different aspects:
    - Classical ML provides robust, interpretable baseline performance
    - BERT captures complex linguistic patterns and context
    - Computer vision adds visual information invisible to text-only models
    
    **Risk Mitigation**: Ensemble approach reduces dependence on any single model's weaknesses
    
    **Scalability**: Production-ready architecture suitable for real-world deployment
    
    ### Expected Outcomes
    - **Performance**: Target exceeding official benchmark performance
    - **Robustness**: Improved handling of missing descriptions and noisy data
    - **Business Value**: Automated classification with measurable accuracy improvements
    
    *Detailed technical methodology and architecture covered in next section...*
    """)

with metrics_tab:
    st.markdown("""
    ## Evaluation Framework
    
    ### Primary Metric
    **Weighted F1-Score**- Accounts for class imbalance in the 27-category classification
    
    ```python
    from sklearn.metrics import f1_score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    ```
    
    ### Experimental Methodology
    
    **Research Focus**: *Training set exploration and methodology development*
    - **Scope**: Comprehensive analysis of 84,916 training samples
    - **Validation**: Rigorous train/validation splits with proper methodology
    - **Comparison**: Individual model performance vs. ensemble effectiveness
    
    **Performance Analysis**
    - **Individual model evaluation**: SVM, BERT, VGG16 standalone performance
    - **Ensemble optimization**: Systematic weight tuning and combination strategies
    - **Hypothesis testing**: Validation of H1-H7 research questions
    
    ### Success Criteria
    
    **Methodology Validation**
    - Demonstrate multimodal approach effectiveness
    - Show complementary value of different model types
    - Establish reproducible ensemble framework
    
    **Research Contributions**
    - Comprehensive preprocessing pipeline for e-commerce data
    - Systematic multimodal ensemble methodology
    - Production-ready classification system
    
    **Practical Impact**
    - End-to-end solution from raw data to predictions
    - Scalable architecture for real-world deployment
    - Interactive demonstration of multimodal classification
    
    ---
    
    ### Academic Context
    *This project serves as the capstone for the Data Science module, demonstrating end-to-end machine learning pipeline development from exploratory data analysis through multimodal model deployment.*
    
    **GitHub Repository**: [Data_Science_Classification_of_Rakuten_Challenge_Multimodal_Soft_Vote](https://github.com/Wilsbert12/Data_Science_Classification_of_Rakuten_Challenge_Multimodal_Soft_Vote)
    """)

# Pagination and footer
add_pagination_and_footer("2_Project_Outline.py")