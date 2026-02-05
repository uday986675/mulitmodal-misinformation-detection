"""
Streamlit Deployment App
========================
Web interface for multimodal misinformation detection model.
Allows users to input text and images for real-time predictions.
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.text_encoder import TextEncoder
    from models.image_encoder import ImageEncoder
    from models.multimodal_fusion import MultimodalFusion
    from models.classifier import MultimodalClassifier, CompleteMultimodalModel
    from data.preprocess_text import TextPreprocessor
    from data.preprocess_image import ImagePreprocessor
    from inference.predict import Predictor
    from utils.config import Config, DEFAULT_CONFIG
    from utils.logger import Logger
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-container {
        padding: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fake-news {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
    }
    .real-news {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION & CACHING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model from checkpoint."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build model components with config
        config = DEFAULT_CONFIG
        
        text_encoder = TextEncoder(
            model_name=config.data.text_model,
            hidden_dim=config.model.text_hidden_dim,
            dropout=config.model.text_dropout,
            freeze_backbone=config.model.text_freeze_backbone,
        )
        
        image_encoder = ImageEncoder(
            model_name=config.model.image_encoder_type,
            output_dim=config.model.image_hidden_dim,
            dropout=config.model.image_dropout,
        )
        
        fusion = MultimodalFusion(
            text_dim=config.model.text_hidden_dim,
            image_dim=config.model.image_hidden_dim,
            hidden_dim=config.model.fusion_hidden_dim,
            dropout=config.model.text_dropout,
            fusion_method=config.model.fusion_type,
        )
        
        classifier = MultimodalClassifier(
            input_dim=config.model.fusion_hidden_dim,
            hidden_dim=config.model.classifier_hidden_dim,
            num_classes=config.model.output_dim,
            dropout=config.model.classifier_dropout,
        )
        
        # Complete model
        model = CompleteMultimodalModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            fusion_module=fusion,
            classifier=classifier,
        )
        
        # Try to load checkpoint if it exists
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "final_model.pt"
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✅ Model checkpoint loaded from {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {str(e)}. Using untrained model.")
        else:
            print(f"⚠️ Model checkpoint not found at {checkpoint_path}. Using untrained model.")
        
        model = model.to(device)
        model.eval()
        
        return model, device
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


@st.cache_resource
def load_preprocessors():
    """Load text and image preprocessors."""
    try:
        # Create config object properly
        config = DEFAULT_CONFIG
        
        # Initialize preprocessors with proper parameters from config
        text_preprocessor = TextPreprocessor(
            model_name=config.data.text_model,
            max_length=config.data.text_max_length
        )
        image_preprocessor = ImagePreprocessor(
            img_size=config.data.image_size,
            augment=config.data.image_augmentation,
            normalize=True
        )
        return text_preprocessor, image_preprocessor
    
    except Exception as e:
        st.error(f"❌ Error loading preprocessors: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


@st.cache_resource
def load_predictor():
    """Load the full prediction pipeline."""
    model_data = load_model()
    text_prep, image_prep = load_preprocessors()
    
    if model_data is None or text_prep is None:
        return None
    
    model, device = model_data
    predictor = Predictor(
        model=model,
        text_preprocessor=text_prep,
        image_preprocessor=image_prep,
        device=device,
        logger=Logger()
    )
    
    return predictor


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_misinformation(text: str, image_path=None):
    """
    Predict whether content is real or fake.
    
    Args:
        text: Social media post text
        image_path: Optional path to image file
        
    Returns:
        Dictionary with prediction results
    """
    predictor = load_predictor()
    
    if predictor is None:
        return None
    
    try:
        result = predictor.predict_single(
            text=text,
            image_path=image_path,
            return_embeddings=False
        )
        return result
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_prediction_result(result):
    """Display prediction result with visualization."""
    if result is None:
        return
    
    prediction = result.get('prediction', -1)
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', [0, 0])
    
    # Determine classification
    is_fake = prediction == 1
    label = "🚨 FAKE NEWS" if is_fake else "✅ REAL NEWS"
    css_class = "fake-news" if is_fake else "real-news"
    
    # Display prediction box
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"<div class='prediction-box {css_class}'>"
                   f"<h2>{label}</h2>"
                   f"<div class='confidence-text'>Confidence: {confidence:.1%}</div>"
                   f"</div>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Fake Probability", f"{probabilities[1]:.1%}")
    
    with col3:
        st.metric("Real Probability", f"{probabilities[0]:.1%}")
    
    # Confidence gauge
    st.write("### Prediction Confidence")
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(probabilities[1], text=f"Fake: {probabilities[1]:.1%}")
    
    with col2:
        st.progress(probabilities[0], text=f"Real: {probabilities[0]:.1%}")


def display_info_section():
    """Display information about the model."""
    with st.expander("📋 About This Model", expanded=False):
        st.markdown("""
        ### Multimodal Misinformation Detection System
        
        This model detects fake news and misinformation in social media using:
        
        **🧠 Model Architecture:**
        - **Text Encoder**: DistilBERT (768-dim embeddings)
        - **Image Encoder**: EfficientNet-B0 (1280-dim embeddings)
        - **Fusion Method**: Advanced multimodal fusion strategies
        - **Classification**: Binary classifier with confidence scores
        
        **📊 Training Details:**
        - **Datasets**: GossipCop + PolitiFact
        - **Loss Function**: Focal Loss + Label Smoothing
        - **Metrics**: F1, ROC-AUC, Accuracy
        
        **⚠️ Limitations:**
        - Works best with English content
        - Requires both text and image (text-only also supported)
        - Predictions may vary based on content quality
        
        **🔐 Privacy:**
        - All processing happens locally
        - No data is sent to external servers
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🔍 Misinformation Detection")
    st.subheader("Detect fake news using multimodal AI analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Select Input Mode",
            ["📝 Text Only", "📝 Text + 🖼️ Image"],
            help="Choose whether to use text alone or both text and image"
        )
        
        show_info = st.checkbox("Show Model Information", value=True)
        
        device_info = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Running on: {device_info}")
        
        if torch.cuda.is_available():
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Display model info if selected
    if show_info:
        display_info_section()
    
    # Main input section
    st.divider()
    st.write("### Enter Content to Analyze")
    
    # Text input
    text_input = st.text_area(
        "📝 Social Media Post Text",
        placeholder="Paste the text content you want to analyze...",
        height=150,
        max_chars=5000,
        help="Enter the text of the social media post or news article"
    )
    
    image_input = None
    
    # Image input (if selected)
    if mode == "📝 Text + 🖼️ Image":
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "🖼️ Upload Image",
                type=["jpg", "jpeg", "png", "bmp", "gif"],
                help="Upload an image associated with the post"
            )
            
            if uploaded_file is not None:
                image_input = uploaded_file
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.write("**Or**")
            image_url = st.text_input(
                "🌐 Image URL",
                placeholder="https://example.com/image.jpg",
                help="Provide a URL to an image"
            )
    
    # Prediction button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔍 Analyze", use_container_width=True, type="primary"):
            if not text_input.strip():
                st.error("❌ Please enter some text to analyze")
            else:
                # Show loading state
                with st.spinner("🔄 Analyzing content..."):
                    result = predict_misinformation(text_input, image_input)
                    
                    if result is not None:
                        st.success("✅ Analysis complete!")
                        display_prediction_result(result)
                    else:
                        st.error("❌ Prediction failed. Please try again.")
    
    # Example section
    st.divider()
    
    with st.expander("📌 Example Posts", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Example Real News:**")
            example_real = "Scientists discover new species of deep-sea fish in Mariana Trench. The discovery was made during a recent research expedition by marine biologists."
            if st.button("📋 Use This Text (Real)", use_container_width=True):
                st.session_state.text_input = example_real
                st.rerun()
        
        with col2:
            st.write("**Example Fake News:**")
            example_fake = "Breaking: World's richest person announces giving away all their wealth. This is not real but demonstrates satirical news."
            if st.button("📋 Use This Text (Fake)", use_container_width=True):
                st.session_state.text_input = example_fake
                st.rerun()
    
    # Footer
    st.divider()
    st.caption("🔐 **Privacy**: All processing happens locally. No data is sent to external servers.")
    st.caption("📊 **Model**: Multimodal Misinformation Detection | Built with PyTorch + Streamlit")


if __name__ == "__main__":
    main()
