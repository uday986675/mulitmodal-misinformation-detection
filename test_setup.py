#!/usr/bin/env python3
"""
Test script to verify Streamlit app setup and model loading.
Run this before deploying to ensure everything works.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")
    try:
        import streamlit as st
        print("  ✅ Streamlit imported")
    except ImportError:
        print("  ❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    try:
        from models.text_encoder import TextEncoder
        from models.image_encoder import ImageEncoder
        from models.multimodal_fusion import MultimodalFusion
        from models.classifier import CompleteMultimodalModel
        from data.preprocess_text import TextPreprocessor
        from data.preprocess_image import ImagePreprocessor
        from inference.predict import Predictor
        from utils.config import DEFAULT_CONFIG
        print("  ✅ All model modules imported")
    except ImportError as e:
        print(f"  ❌ Failed to import model modules: {e}")
        return False
    
    return True


def test_checkpoint():
    """Test that checkpoint exists."""
    print("\n🔍 Testing checkpoint...")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "final_model.pt"
    
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Checkpoint found: {checkpoint_path}")
        print(f"     Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"  ❌ Checkpoint not found at {checkpoint_path}")
        print("     Make sure 'final_model.pt' is in the 'checkpoints/' folder")
        return False


def test_device():
    """Test device availability."""
    print("\n🔍 Testing device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ✅ Using device: {device.upper()}")
    
    if device == "cuda":
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return True


def test_model_loading():
    """Test that model can be loaded."""
    print("\n🔍 Testing model loading...")
    
    try:
        from models.classifier import CompleteMultimodalModel
        from utils.config import DEFAULT_CONFIG
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = DEFAULT_CONFIG
        
        model = CompleteMultimodalModel(config)
        print(f"  ✅ Model instantiated: {model.__class__.__name__}")
        
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "final_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("  ✅ Checkpoint loaded successfully")
            
            model = model.to(device)
            model.eval()
            print("  ✅ Model moved to device and set to eval mode")
            return True
        else:
            print("  ⚠️  Model can be instantiated but checkpoint not found")
            return True
    
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False


def test_preprocessors():
    """Test that preprocessors can be initialized."""
    print("\n🔍 Testing preprocessors...")
    
    try:
        from data.preprocess_text import TextPreprocessor
        from data.preprocess_image import ImagePreprocessor
        from utils.config import DEFAULT_CONFIG
        
        config = DEFAULT_CONFIG
        text_prep = TextPreprocessor(config)
        print("  ✅ TextPreprocessor initialized")
        
        image_prep = ImagePreprocessor(config)
        print("  ✅ ImagePreprocessor initialized")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Failed to initialize preprocessors: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  🚀 Streamlit App Setup Verification")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Checkpoint", test_checkpoint),
        ("Device", test_device),
        ("Preprocessors", test_preprocessors),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("  📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ✅ All tests passed! Ready to deploy.")
        print("\n  Run the app with:")
        print("     streamlit run app.py")
        print("\n" + "="*60 + "\n")
        return 0
    else:
        print("\n  ❌ Some tests failed. Fix issues before deploying.")
        print("\n" + "="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
