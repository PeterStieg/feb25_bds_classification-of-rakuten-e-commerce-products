# Methodology
import streamlit as st
import sys
import os
from streamlit_mermaid import st_mermaid

# Add project root to path to import streamlit_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Methodology",
    page_icon="streamlit/assets/images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(3 / 7)
st.title("Technical Methodology")
st.sidebar.header(":material/model_training: Methodology")
st.sidebar.image("streamlit/assets/images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Methodology overview
st.markdown("""
## Multimodal Ensemble Architecture

Our approach leverages the **complementary strengths**of three distinct machine learning paradigms to create a robust multimodal classification system that exceeds official benchmarks.
""")

# Load and display methodology diagram
try:
    with open("streamlit/assets/methodology.mmd", "r") as file:
        methodology_mermaid = file.read()
    
    st.markdown("###  Complete Pipeline Architecture")
    st_mermaid(methodology_mermaid, height="auto")
except FileNotFoundError:
    st.warning("Methodology diagram not found. Please ensure methodology.mmd is in streamlit/assets/")

st.markdown("---")

# Technical methodology tabs
strategy_tab, models_tab, ensemble_tab, optimization_tab = st.tabs(
    ["Ensemble Strategy", "Individual Models", "Ensemble Integration", "Technical Innovations"]
)

with strategy_tab:
    st.markdown("""
    ## Why Multimodal Ensemble?
    
    ### Core Technical Hypothesis
    **Different model architectures capture fundamentally different aspects of product information that are complementary rather than redundant.**
    
    ### Strategic Advantages
    
    **Complementary Information Processing**
    - **Classical ML**: Robust statistical pattern recognition, interpretable features
    - **Deep Learning**: Complex contextual understanding, semantic relationships  
    - **Computer Vision**: Visual product characteristics invisible to text-only models
    
    **Risk Mitigation Through Diversity**
    - **Model diversity**: Reduces overfitting to specific data patterns
    - **Failure independence**: Different models fail on different examples
    - **Robust performance**: Ensemble maintains accuracy even when individual models struggle
    
    **Handling Real-World E-commerce Challenges**
    - **Missing descriptions**: CV and title-only processing provide backup
    - **Multilingual content**: BERT + translation handles language diversity
    - **Noisy merchant data**: Classical ML provides robust baseline performance
    
    ### Ensemble Composition Rationale
    
    **Classical ML Foundation**
    - Provides interpretable, robust baseline
    - Excellent performance on clean, structured text
    - Computationally efficient for production deployment
    
    **BERT Context Understanding**
    - Captures semantic meaning and context
    - Handles French language nuances natively
    - Strong performance on complex product descriptions
    
    **Visual Information Complement**
    - Adds unique visual product characteristics
    - Provides backup for text-poor products
    - Enhances classification confidence through multimodal confirmation
    """)

with models_tab:
    st.markdown("""
    ## Individual Model Architecture Details
    
    ###  Classical Machine Learning Pipeline
    
    **Algorithm Evaluation: Multiple Approaches**
    - **Algorithms tested**: Support Vector Machine, Logistic Regression, XGBoost, Random Forest
    - **Architecture**: TF-IDF vectorization with systematic hyperparameter optimization
    - **Rationale**: Robust to noisy e-commerce text, interpretable features
    - **Optimization**: Grid search with 3-fold cross-validation for best algorithm selection
    
    **Text Processing Strategy:**
    - Combined text preprocessing (designation + description)
    - French-specific language handling
    - Translation integration for multilingual content
    - Model-specific preprocessing pipelines
    
    *Detailed preprocessing pipelines shown in Data & Preprocessing section...*
    
    **Key Strengths:**
    - Robust to noisy e-commerce text
    - Interpretable feature importance
    - Fast training and inference
    - Excellent baseline performance
    
    ---
    
    ###  Natural Language Processing (CamemBERT)
    
    **Model Selection: CamemBERT-base**
    - **Architecture**: French BERT variant fine-tuned for classification
    - **Rationale**: Native French language understanding, contextual semantics
    - **Optimization**: Learning rate scheduling, early stopping
    
    **Processing Strategy:**
    - Minimal preprocessing for BERT compatibility
    - Native French accent handling via CamemBERT tokenizer
    - Sequence length optimization
    - Special token integration for classification
    
    *Detailed text processing shown in Data & Preprocessing section...*
    
    **Key Strengths:**
    - Native French language understanding
    - Contextual semantic analysis
    - Transfer learning from large corpus
    - State-of-the-art NLP performance
    
    ---
    
    ###  Computer Vision (VGG16)
    
    **Model Selection: VGG16 with Transfer Learning**
    - **Architecture**: Pre-trained VGG16 with frozen feature layers + custom classification head
    - **Rationale**: Visual product characteristics, backup for text-poor products
    - **Optimization**: Transfer learning with frozen feature extraction layers
    
    **Image Processing Strategy:**
    - Automated product focus detection
    - Intelligent cropping and resizing
    - CNN-optimized preprocessing pipeline
    - Data augmentation for training robustness
    
    *Detailed image processing pipeline shown in Data & Preprocessing section...*
    
    **Key Strengths:**
    - Visual product characteristic analysis
    - Robust to text quality issues
    - Complementary information source
    - Production-ready image processing
    """)

with ensemble_tab:
    st.markdown("""
    ## Ensemble Integration Methodology
    
    ### Soft Voting Strategy
    
    **Why Soft Voting Over Hard Voting?**
    - **Probability information**: Utilizes prediction confidence, not just final class
    - **Weighted combination**: Allows optimization of individual model contributions
    - **Smooth decision boundaries**: Reduces classification noise from conflicting predictions
    
    ### Mathematical Framework
    
    ```python
    # Soft voting ensemble prediction
    ensemble_proba = (w1 * svm_proba + 
                     w2 * bert_proba + 
                     w3 * vgg16_proba)
    
    final_prediction = argmax(ensemble_proba)
    ```
    
    ### Ensemble Integration Framework
    
    **Soft Voting Strategy**
    - **Probability information**: Utilizes prediction confidence, not just final class
    - **Weighted combination**: Allows optimization of individual model contributions
    - **Smooth decision boundaries**: Reduces classification noise from conflicting predictions
    
    **Mathematical Framework**
    ```python
    # Soft voting ensemble prediction
    ensemble_proba = (w1 * model1_proba + 
                     w2 * model2_proba + 
                     w3 * model3_proba)
    
    final_prediction = argmax(ensemble_proba)
    ```
    
    **Optimization Framework:**
    ```python
    weight_optimization = {
        'method': 'Selected configuration testing',
        'configurations': [
            'Equal weights (0.33, 0.33, 0.34)',
            'Text-heavy (0.4, 0.4, 0.2)', 
            'BERT-heavy (0.3, 0.5, 0.2)',
            'SVM-heavy (0.5, 0.3, 0.2)'
        ],
        'validation_method': 'Clean train/val split',
        'optimization_metric': 'Weighted F1-score'
    }
    ```
    
    *Optimal weights and performance results presented in Models & Results section...*
    
    ### Validation Framework
    
    **Evaluation Strategy:**
    - **Clean train/validation split**: Proper methodology without data leakage
    - **Consistent random state**: All models use identical data partitions
    - **Metric focus**: Weighted F1-score for class imbalance handling
    - **Benchmark comparison**: Performance vs official Rakuten challenge results
    
    *Detailed performance results and model comparisons presented in Models & Results section...*
    """)

with optimization_tab:
    st.markdown("""
    ## Technical Framework & Architecture
    
    ### Data Processing Strategy
    
    **Multilingual Content Handling:**
    - Language detection and translation workflow
    - Pre-processed French translations for consistent model input
    - Model-specific text integration strategies
    
    **Image Processing Framework:**
    - Automated product focus detection and cropping
    - CNN-optimized preprocessing pipeline
    - Storage-efficient processing with quality control
    
    *Detailed preprocessing pipelines and data exploration shown in Data & Preprocessing section...*
    
    ### Model Integration Architecture
    
    **Consistent Data Splitting:**
    - Global reproducibility framework with fixed random states
    - Stratified splitting ensuring fair model comparison
    - No data leakage between train/validation/test sets
    
    **Production Readiness:**
    - Modular architecture with independent model loading
    - Systematic model artifact storage and versioning
    - API-ready inference pipeline for web service deployment
    """)

# Pagination and footer
add_pagination_and_footer("3_Methodology.py")