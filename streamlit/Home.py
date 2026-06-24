# Home.py
import streamlit as st
import sys
import os

# Add parent directory to path to import streamlit_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Rakuten: Multimodal Product Classification",
    page_icon="streamlit/assets/images/logos/rakuten-favicon.ico",
    layout="wide",
)

# Add Open Graph meta tags
st.markdown(
    """
    <head>
        <meta property="og:description" content="Multimodal machine learning solution for e-commerce product classification combining text and image data. Built as capstone project for DataScientist training program FEB25 BDS." />
    </head>
""",
    unsafe_allow_html=True,
)

st.progress(0 / 7)
st.title("Multimodal Product Classification")

st.sidebar.header(":material/home: Home")
st.sidebar.image("streamlit/assets/images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Hero section
st.markdown("""
## E-Commerce Product Classification Using Advanced Machine Learning

**Automated multimodal classification system**for Rakuten France's marketplace, combining text analysis and computer vision to categorize products at scale.

---
""")

# Key highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ###  Challenge
    **Large-scale product categorization**
    - 84,916 training products
    - 27 product categories  
    - Multimodal data (text + images)
    - Real-world e-commerce complexity
    """)

with col2:
    st.markdown("""
    ###  Solution
    **Ensemble learning approach**
    - Classical ML (SVM + TF-IDF)
    - Deep Learning (CamemBERT)
    - Computer Vision (VGG16)
    - Optimized multimodal fusion
    """)

with col3:
    st.markdown("""
    ###  Innovation
    **End-to-end pipeline**
    - Comprehensive preprocessing
    - Language detection & translation
    - Image enhancement & cropping
    - Production-ready deployment
    """)

st.markdown("---")

# Project overview
overview_col1, overview_col2 = st.columns([2, 1])

with overview_col1:
    st.markdown("""
    ### Project Overview
    
    This project addresses **Rakuten's challenge**of accurately categorizing products in their French marketplace using both textual descriptions and product images. Our approach leverages the complementary strengths of different machine learning paradigms:
    
    - **Classical Machine Learning**for robust statistical pattern recognition
    - **Natural Language Processing**for contextual text understanding  
    - **Computer Vision**for visual product analysis
    
    **Business Impact**: Automated product categorization enables improved search functionality, personalized recommendations, and scalable catalog management for over 1.3 billion Rakuten users worldwide.
    
    **Technical Achievement**: Successfully developed and deployed a multimodal ensemble classifier that demonstrates effective fusion of text and image modalities for e-commerce applications.
    """)

with overview_col2:
    st.info("""
    **Dataset Characteristics**
    
    **Training Data**: 84,916 products  
    **Categories**: 27 product type codes  
    **Languages**: French, German, multilingual  
    **Missing Data**: ~35% products lack descriptions  
    **Modalities**: Text + Images  
    
    **Academic Context**
    
    Capstone project for **Data Science**module  
    **Cohort**: FEB25 BDS  
    **Focus**: End-to-end ML pipeline development  
    """)

st.markdown("---")

# Navigation guide
st.markdown("""
###  Presentation Guide

Use the **sidebar navigation**or **pagination buttons**below to explore:

1. **Team Presentation**- Meet the team and individual contributions
2. **Project Outline**- Business context, technical challenge, and our approach  
3. **Methodology**- Detailed technical methodology and ensemble architecture
4. **Data & Preprocessing**- Dataset analysis, hypotheses, and preprocessing pipelines
5. **Models & Results**- Individual model performance and ensemble optimization
6. **Live Demo**- Interactive multimodal classifier demonstration
7. **Thank You**- Project conclusions and team contact information

---

**Highlight**: Don't miss the **Live Demo**(page 6) where you can test the multimodal classifier with your own product images and descriptions!
""")

# Project links and context
st.markdown("""
###  Project Resources

**GitHub Repository**: [Data_Science_Classification_of_Rakuten_Challenge_Multimodal_Soft_Vote](https://github.com/Wilsbert12/Data_Science_Classification_of_Rakuten_Challenge_Multimodal_Soft_Vote)

**Challenge Source**: [Rakuten Data Challenge - ENS](https://challengedata.ens.fr/challenges/35)

**Team**: Peter Stieg, Robert Wilson, Thomas Borer | **Program**: DataScientist FEB25 BDS
""")

# Pagination and footer
add_pagination_and_footer("Home.py")# Updated Tue Aug 12 02:46:28 CEST 2025
