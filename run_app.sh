#!/bin/bash
# Run Streamlit App - Linux/Mac Version
# Simple script to start the Streamlit application

echo "================================"
echo "🚀 Starting Misinformation Detector"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if Streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Streamlit not found. Installing..."
    pip install streamlit==1.28.0
    echo ""
fi

# Check if model checkpoint exists
if [ ! -f "checkpoints/final_model.pt" ]; then
    echo "⚠️  Warning: Model checkpoint not found at checkpoints/final_model.pt"
    echo ""
fi

echo "🔄 Starting Streamlit app..."
echo ""
echo "The app will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

# Run Streamlit app
streamlit run app.py
