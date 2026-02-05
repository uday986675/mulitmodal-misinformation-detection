# 🚀 Streamlit Deployment - Complete Guide

## Overview

This directory contains a fully functional Streamlit web application for deploying your **Multimodal Misinformation Detection** model. The app provides an intuitive interface for analyzing social media posts and detecting fake news using both text and image analysis.

---

## 📁 New Files Created

```
.
├── app.py                          ← Main Streamlit web app
├── requirements_streamlit.txt      ← Full dependencies with Streamlit
├── test_setup.py                   ← Verification script
├── Dockerfile                      ← Docker containerization
├── docker-compose.yml              ← Docker Compose orchestration
├── .streamlit/
│   └── config.toml                 ← Streamlit configuration
├── DEPLOYMENT_GUIDE.md             ← Detailed deployment guide
└── QUICKSTART.md                   ← Quick start instructions (3 mins)
```

---

## ⚡ Quick Start (30 seconds)

### Step 1: Install Streamlit
```bash
pip install streamlit
```

### Step 2: Run the App
```bash
streamlit run app.py
```

### Step 3: Open in Browser
Visit: **http://localhost:8501**

That's it! 🎉

---

## 🎯 Features

### 🧠 What It Does
- **Text Analysis**: Analyzes social media posts to detect misinformation
- **Image Support**: Optional image upload for multimodal analysis
- **Real-time Predictions**: Instant confidence scores with visual feedback
- **Privacy**: All processing happens locally - no data sent to cloud

### 🎨 User Interface
- Clean, intuitive design with Streamlit
- Text input area for post content
- Image upload with preview
- Color-coded results (🚨 Fake / ✅ Real)
- Confidence percentage and probability distribution
- Example posts for testing

### ⚙️ Technical Features
- **GPU Support**: Automatically detects and uses CUDA
- **Model Caching**: Fast startup after first run
- **Configuration UI**: Adjust settings from sidebar
- **Device Info**: Shows GPU/CPU status

---

## 🖥️ System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 4 GB
- **Disk**: 2 GB (+ 280 MB for model checkpoint)

### Recommended
- **Python**: 3.10+
- **RAM**: 8 GB+
- **GPU**: NVIDIA GPU with CUDA 11.8+
- **Disk**: 5 GB

---

## 📦 Installation

### Method 1: Using requirements_streamlit.txt (Recommended)
```bash
pip install -r requirements_streamlit.txt
```

### Method 2: Install individual packages
```bash
# Core dependencies
pip install torch torchvision transformers

# Web framework
pip install streamlit

# Supporting libraries
pip install numpy pandas scikit-learn Pillow
```

---

## 🚀 Running the App

### Local Development
```bash
streamlit run app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Custom Configuration
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Headless mode (for servers)
streamlit run app.py --server.headless true

# Disable browser auto-open
streamlit run app.py --logger.level=warning
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t misinformation-detector:latest .
```

### Run Container
```bash
docker run -p 8501:8501 misinformation-detector:latest
```

### Using Docker Compose (Easiest)
```bash
docker-compose up
```

Access at: **http://localhost:8501**

**Stop container:**
```bash
docker-compose down
```

---

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (⭐ Easiest)

1. Push code to GitHub
2. Go to: https://streamlit.io/cloud
3. Click "New app" and select your repo
4. Choose `app.py` as the main file
5. Click Deploy!

**Cost**: Free tier available  
**Time**: 2 minutes

### Option 2: AWS EC2

```bash
# Launch t3.medium instance with Ubuntu 22.04

# SSH into instance
ssh -i key.pem ubuntu@your-instance

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements_streamlit.txt

# Run app
streamlit run app.py --server.port 80

# Keep running (using nohup)
nohup streamlit run app.py --server.port 80 > app.log 2>&1 &
```

**Cost**: $0.05-0.20/hour (t3.medium)  
**Time**: 10 minutes

### Option 3: Google Cloud Run

```bash
# Create app.yaml
gcloud run deploy misinformation-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --port 8501

# Your app is live at the provided URL
```

**Cost**: Free tier (some requests)  
**Time**: 5 minutes

### Option 4: Railway, Heroku, Render
Similar to Streamlit Cloud - push to GitHub and deploy!

---

## 🧪 Verification

### Run Setup Test
```bash
python test_setup.py
```

**Expected output:**
```
============================================================
  🚀 Streamlit App Setup Verification
============================================================

🔍 Testing imports...
  ✅ Streamlit imported

🔍 Testing checkpoint...
  ✅ Checkpoint found: checkpoints/final_model.pt
     Size: 279.47 MB

🔍 Testing device...
  ✅ Using device: CUDA
     GPU: NVIDIA GeForce RTX 3090

🔍 Testing preprocessors...
  ✅ TextPreprocessor initialized

🔍 Testing model loading...
  ✅ Checkpoint loaded successfully

============================================================
  📊 Test Summary
============================================================
  ✅ PASS: Imports
  ✅ PASS: Checkpoint
  ✅ PASS: Device
  ✅ PASS: Preprocessors
  ✅ PASS: Model Loading

  Result: 5/5 tests passed

  ✅ All tests passed! Ready to deploy.
============================================================
```

---

## 📊 App Usage

### 1. Select Input Mode
- **Text Only**: Faster analysis using only text
- **Text + Image**: More accurate with multimodal fusion

### 2. Enter Content
- Paste social media post text (up to 5000 characters)
- Optionally upload associated image

### 3. Click Analyze
- Real-time prediction with confidence score
- Visual indicators (🚨 Fake / ✅ Real)
- Probability distribution chart

### 4. Review Results
- **Prediction**: Binary classification (Real/Fake)
- **Confidence**: Likelihood score (0-100%)
- **Probabilities**: Detailed breakdown per class

---

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)

**Key Settings:**
```toml
[server]
port = 8501                    # Port number
headless = true               # No browser auto-open
maxUploadSize = 200           # Max upload 200 MB

[theme]
primaryColor = "#1f77b4"      # Blue accent
backgroundColor = "#ffffff"   # White background
```

**Modify for:**
- Different port
- Custom colors
- Upload limits
- Logging level

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port 8501 already in use** | `streamlit run app.py --server.port 8502` |
| **Model not found** | Ensure `checkpoints/final_model.pt` exists |
| **"Module not found" errors** | Run `pip install -r requirements_streamlit.txt` |
| **App is very slow** | Use GPU or upgrade instance specs |
| **Out of memory** | Reduce batch size or restart Streamlit |
| **CUDA not detected** | Install `torch[cuda]` matching your CUDA version |
| **Image upload fails** | Ensure Pillow is installed: `pip install Pillow` |

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Startup Time** | 3-5 seconds (cached model) |
| **Inference Time (CPU)** | 2-3 seconds |
| **Inference Time (GPU)** | 0.5-1 second |
| **Memory Usage** | 2-3 GB |
| **Max Concurrent Users** | 1 (single server) |

**For multiple users**: Deploy multiple instances or use load balancer.

---

## 🔐 Security Considerations

### Data Privacy
✅ All processing is **local** - no cloud uploads  
✅ No external API calls  
✅ Model and inputs stay on server  

### Production Deployment
- Use HTTPS (let's encrypt)
- Add authentication if needed
- Set rate limiting
- Monitor resource usage
- Keep dependencies updated

---

## 📚 Project Structure

```
Multimodal Misinformation Detection in Noisy Social Streams/
├── app.py                    # Streamlit web app (START HERE)
├── main.py                   # Training/inference CLI
├── requirements.txt          # Core dependencies
├── requirements_streamlit.txt # With Streamlit
├── test_setup.py             # Verification script
│
├── models/                   # Model architecture
│   ├── text_encoder.py
│   ├── image_encoder.py
│   ├── multimodal_fusion.py
│   └── classifier.py
│
├── data/                     # Preprocessing
│   ├── preprocess_text.py
│   └── preprocess_image.py
│
├── inference/                # Prediction pipeline
│   └── predict.py
│
├── checkpoints/              # Trained models
│   └── final_model.pt        # 279 MB model checkpoint
│
├── .streamlit/
│   └── config.toml          # Streamlit settings
│
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker orchestration
├── QUICKSTART.md             # 3-minute setup
├── DEPLOYMENT_GUIDE.md       # Detailed guide
└── STREAMLIT_README.md       # This file
```

---

## 🎓 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs
- **Docker Docs**: https://docs.docker.com
- **Model Details**: See [README.md](README.md)

---

## 💡 Advanced Usage

### Custom Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"           # Red
backgroundColor = "#1a1a1a"        # Dark
textColor = "#ffffff"              # White
```

### Enable HTTPS
```bash
streamlit run app.py \
  --logger.level=warning \
  --server.sslCertFile=/path/to/cert.pem \
  --server.sslKeyFile=/path/to/key.pem
```

### API Access
Use the Predictor class directly for programmatic access:
```python
from inference.predict import Predictor

predictor = Predictor(model, text_prep, image_prep)
result = predictor.predict_single("Your text", "image.jpg")
```

---

## 📞 Support

**Issues?**
1. Check model checkpoint exists
2. Verify all dependencies: `pip install -r requirements_streamlit.txt`
3. Run verification: `python test_setup.py`
4. Check logs: `streamlit run app.py --logger.level=debug`

**Questions?**
- Streamlit: https://discuss.streamlit.io
- Model: See [README.md](README.md)
- Deployment: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📝 License

This project follows the original project's license.

---

**Happy deploying! 🎉**

*Created with ❤️ using Streamlit & PyTorch*
