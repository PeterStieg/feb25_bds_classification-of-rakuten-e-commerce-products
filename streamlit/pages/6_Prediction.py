# Prediction
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamlit_utils import add_pagination_and_footer
from utils.text_utils import text_cleaner
from utils.image_utils import preprocess_image
from api.ensemble_service import RakutenEnsembleClassifier

st.set_page_config(
    page_title="FEB25 BDS // Prediction",
    page_icon="streamlit/assets/images/logos/rakuten-favicon.ico",
    layout="wide",
)

# Load category mapping
@st.cache_data
def load_category_mapping():
    try:
        import json
        with open('data/prdtypecode_to_category_name.json', 'r') as f:
            return json.load(f)
    except:
        return {}

category_mapping = load_category_mapping()

# Load ensemble model with lazy loading
@st.cache_resource
def load_ensemble():
    """Load the ensemble classifier with caching - only when needed"""
    classifier = RakutenEnsembleClassifier()
    classifier.load_models()
    return classifier

def get_classifier():
    """Get classifier with lazy loading and progress indicator"""
    if 'classifier' not in st.session_state:
        with st.spinner(" Loading ensemble models... (this may take up to 30 seconds)"):
            try:
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                st.write("Loading SVM + TF-IDF models...")
                
                progress_bar.progress(50)
                st.write("Loading BERT model...")
                
                progress_bar.progress(75)
                st.write("Loading VGG16 model...")
                
                classifier = load_ensemble()
                
                progress_bar.progress(100)
                st.success(" Ensemble models loaded successfully!")
                st.session_state.classifier = classifier
                
            except Exception as e:
                st.error(f" Failed to load models: {e}")
                return None
                
    return st.session_state.classifier

# Demo samples data
DEMO_SAMPLES = [
    {
        'sample_index': 48840,
        'imageid': 1201825641,
        'productid': 3001560613,
        'category': 'Books & Literature',
        'prdtypecode': 10,
        'title': 'Disney Recital Suites: Arr. Phillip Keveren The Phillip Keveren Series Piano Solo',
        'description': 'Piano ou Clavier / Keyboard or Piano - 20 Disney favorites arranged by Phillip Keveren for piano solo',
        'image_path': 'streamlit/assets/images/samples/sample_48840.jpg'
    },
    {
        'sample_index': 869,
        'imageid': 1225835982,
        'productid': 3593005420,
        'category': 'Furniture & Home',
        'prdtypecode': 1560,
        'title': 'Baleri Italia Chaise Avec Accoudoirs Kin (Step Cat. B - Tissu Et Acier Chromé Noir)',
        'description': 'Kin Baleri Italia est une système de sièges rembourrés avec mini accoudoirs. Structure en tube d¿acier chromé noir.',
        'image_path': 'streamlit/assets/images/samples/sample_869.jpg'
    },
    {
        'sample_index': 28866,
        'imageid': 1241655964,
        'productid': 3759137530,
        'category': 'Office Supplies',
        'prdtypecode': 2522,
        'title': 'Faber-Castell Stylo-Plume Grip 2010 M Turquoise',
        'description': 'Le stylo-bille et le porte-mine de la collection GRIP 2010 vont épater leurs utilisateurs grâce leur forme triangulaire ergonomique.',
        'image_path': 'streamlit/assets/images/samples/sample_28866.jpg'
    },
    {
        'sample_index': 54112,
        'imageid': 1211802460,
        'productid': 3410001417,
        'category': 'Lighting & Electrical',
        'prdtypecode': 1560,
        'title': '8pcs G95 Forme Antiquité Ronde E27 40w Ac220-240v Ampoules À Incandescence Pour Chambre',
        'description': 'Ampoules à incandescence vintage G95 pour éclairage décoratif avec température de couleur blanc chaud 2800-3500k.',
        'image_path': 'streamlit/assets/images/samples/sample_54112.jpg'
    },
    {
        'sample_index': 72259,
        'imageid': 1298338295,
        'productid': 4138131659,
        'category': 'Toys & Games',
        'prdtypecode': 1281,
        'title': "L'ecriture - Coffret De 6 Jeux Progressifs - 4-7 Ans",
        'description': 'Coffret de 6 jeux évolutifs pour apprendre les premiers graphismes et aller jusqu\'à l\'écriture des mots en lettres cursives.',
        'image_path': 'streamlit/assets/images/samples/sample_72259.jpg'
    },
]

def display_prediction_results(result, category_mapping):
    """Display detailed prediction results"""
    if 'error' in result:
        st.error(f"Prediction failed: {result['error']}")
        return
    
    # Get category name
    prediction_code = result['prediction']
    category_name = category_mapping.get(str(prediction_code), f"Category {prediction_code}")
    
    # Individual model predictions
    st.markdown("###  Individual Model Predictions")
    col1, col2, col3 = st.columns(3)
    
    if 'individual_predictions' in result:
        with col1:
            svm_pred = result['individual_predictions']['svm']
            svm_conf = result['individual_confidences']['svm']
            svm_category = category_mapping.get(str(svm_pred), f"Category {svm_pred}")
            st.metric("SVM Model", svm_category, f"Confidence: {svm_conf:.3f}")
            
        with col2:
            bert_pred = result['individual_predictions']['bert']
            bert_conf = result['individual_confidences']['bert']
            bert_category = category_mapping.get(str(bert_pred), f"Category {bert_pred}")
            st.metric("BERT Model", bert_category, f"Confidence: {bert_conf:.3f}")
            
        with col3:
            vgg16_pred = result['individual_predictions']['vgg16']
            vgg16_conf = result['individual_confidences']['vgg16']
            vgg16_category = category_mapping.get(str(vgg16_pred), f"Category {vgg16_pred}")
            st.metric("VGG16 Model", vgg16_category, f"Confidence: {vgg16_conf:.3f}")
    
    # Ensemble weights
    weights = result.get('ensemble_weights', {})
    st.markdown("###  Ensemble Configuration")
    weight_col1, weight_col2, weight_col3 = st.columns(3)
    with weight_col1:
        st.info(f"SVM Weight: {weights.get('svm', 0.4):.1%}")
    with weight_col2:
        st.info(f"BERT Weight: {weights.get('bert', 0.4):.1%}")
    with weight_col3:
        st.info(f"VGG16 Weight: {weights.get('vgg16', 0.2):.1%}")
    
    # Top predictions
    if 'ensemble_probabilities' in result:
        st.markdown("###  Top 3 Predictions")
        probs = np.array(result['ensemble_probabilities'])
        top_3_indices = np.argsort(probs)[-3:][::-1]
        
        classifier = st.session_state.get('classifier')
        if classifier and hasattr(classifier, 'label_encoder'):
            for i, idx in enumerate(top_3_indices, 1):
                pred_code = classifier.label_encoder.classes_[idx]
                pred_category = category_mapping.get(str(pred_code), f"Category {pred_code}")
                confidence = probs[idx]
                
                if i == 1:
                    st.success(f" **{i}. {pred_category}**- {confidence:.3f}")
                elif i == 2:
                    st.info(f" {i}. {pred_category} - {confidence:.3f}")
                else:
                    st.warning(f" {i}. {pred_category} - {confidence:.3f}")
    
    # Final result
    st.markdown("###  Final Ensemble Prediction")
    st.success(f"**{category_name}**(Confidence: {result['confidence']:.3f})")

st.progress(6 / 7)
st.title("Multimodal Product Classification")
st.sidebar.header(":material/category_search: Prediction")
st.sidebar.image("streamlit/assets/images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

st.markdown("""
**Ensemble Classification Demo**- Combining SVM, BERT, and VGG16 models for product categorization.

 **Performance**: F1 = 0.8727 (exceeds challenge benchmark by 7.6%)  
 **Weights**: SVM (40%) + BERT (40%) + VGG16 (20%)  
 **Categories**: 27 product categories from Rakuten France catalog
""")

# Show loading status
if 'classifier' not in st.session_state:
    st.info(" Click a prediction button below to load the AI models (takes 30-60 seconds)")

# Tabs
tab_sample, tab_upload = st.tabs([" Sample Product", " Upload Your Own"])

with tab_sample:
    st.markdown("### Try a Sample Product")
    st.markdown("See the ensemble in action with curated examples from our training data.")
    
    # Sample selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get current sample (use session state for persistence)
        if 'current_sample_idx' not in st.session_state:
            st.session_state.current_sample_idx = 0
            
        current_sample = DEMO_SAMPLES[st.session_state.current_sample_idx]
        
        # Display sample info
        st.info(f"**Sample {st.session_state.current_sample_idx + 1}/{len(DEMO_SAMPLES)}**: {current_sample['category']}")
        
    with col2:
        if st.button(" Different Sample", type="primary"):
            st.session_state.current_sample_idx = (st.session_state.current_sample_idx + 1) % len(DEMO_SAMPLES)
            st.rerun()
    
    # Display sample
    img_col, text_col = st.columns([1, 2])
    
    with img_col:
        st.markdown("**Product Image**")
        try:
            st.image(current_sample['image_path'], use_container_width=True)
        except:
            st.error("Image not found")
    
    with text_col:
        st.markdown("**Product Information**")
        st.write(f"**Title:**{current_sample['title']}")
        st.write(f"**Description:**{current_sample['description'][:200]}...")
        st.write(f"**True Category:**{current_sample['category']}")
    
    # Prediction button for sample
    if st.button(" Predict Category", key="predict_sample"):
        classifier = get_classifier()
        if classifier is None:
            st.stop()
            
        with st.spinner("Running ensemble prediction..."):
            # Prepare text
            combined_text = f"{current_sample['title']} {current_sample['description']}"
            cleaned_text = text_cleaner(combined_text)
            
            # Make prediction
            result = classifier.predict_single(
                cleaned_text, 
                current_sample['image_path'], 
                return_details=True
            )
            
            st.markdown("---")
            display_prediction_results(result, category_mapping)

with tab_upload:
    st.markdown("### Upload Your Own Product")
    st.markdown("Test the ensemble with your own product images and descriptions.")
    
    # User inputs
    user_title = st.text_input(
        "**Product Title**",
        placeholder="Enter the product title...",
        help="A short, descriptive title for your product"
    )
    
    user_description = st.text_area(
        "**Product Description**", 
        placeholder="Enter a detailed product description...",
        help="Detailed description of the product features and characteristics",
        height=100
    )
    
    uploaded_image = st.file_uploader(
        "**Product Image**",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of your product"
    )
    
    # Prediction button for user input
    predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
    with predict_col2:
        predict_user = st.button(" Predict Category", key="predict_user", type="primary")
    
    if predict_user:
        if not user_title and not user_description:
            st.warning(" Please enter at least a title or description")
        elif not uploaded_image:
            st.warning(" Please upload an image")
        else:
            classifier = get_classifier()
            if classifier is None:
                st.stop()
                
            with st.spinner("Processing your product..."):
                try:
                    # Prepare text
                    combined_text = f"{user_title} {user_description}".strip()
                    cleaned_text = text_cleaner(combined_text)
                    
                    # Load image
                    image = Image.open(uploaded_image).convert("RGB")
                    
                    # Make prediction
                    result = classifier.predict_single(
                        cleaned_text, 
                        image, 
                        return_details=True
                    )
                    
                    st.markdown("---")
                    
                    # Display input summary
                    st.markdown("###  Your Input")
                    input_col1, input_col2 = st.columns([1, 2])
                    
                    with input_col1:
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    with input_col2:
                        st.write(f"**Title:**{user_title}")
                        st.write(f"**Description:**{user_description}")
                        st.write(f"**Processed Text:**{cleaned_text[:150]}...")
                    
                    st.markdown("---")
                    display_prediction_results(result, category_mapping)
                    
                except Exception as e:
                    st.error(f"Error processing your input: {e}")

# Performance info
with st.expander(" Model Performance Information"):
    classifier = st.session_state.get('classifier')
    if classifier:
        try:
            perf_info = classifier.get_performance_info()
            
            st.markdown("###  Ensemble Performance")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("F1-Score (Weighted)", f"{perf_info['model_info']['ensemble_f1_weighted']:.4f}")
            with perf_col2:
                st.metric("F1-Score (Macro)", f"{perf_info['model_info']['ensemble_f1_macro']:.4f}")
            with perf_col3:
                st.metric("vs Benchmark", perf_info['model_info']['performance_vs_benchmark'])
                
            st.markdown("###  Configuration")
            st.json(perf_info['configuration'])
            
        except Exception as e:
            st.error(f"Could not load performance info: {e}")
    else:
        st.info("Load models first by making a prediction to see performance information.")

# Pagination and footer
add_pagination_and_footer("6_Prediction.py")