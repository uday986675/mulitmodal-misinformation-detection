# 🚀 Quick Start Guide - Streamlit Deployment

## Installation & Running (30 seconds)

### 1. Install Streamlit
```bash
pip install streamlit==1.28.0
```

### 2. Run the App
```bash
streamlit run app.py
```

That's it! The app opens at **http://localhost:8501**

---

## What You Get

### 🎯 Main Features
- **Text Analysis**: Paste any text content to detect misinformation
- **Multimodal Analysis**: Upload images with text for better predictions
- **Confidence Scores**: See real/fake probabilities
- **GPU Support**: Automatically uses CUDA if available
- **Privacy**: All processing is local, no cloud uploads

### 🎨 User Interface
```
┌─────────────────────────────────────────┐
│  🔍 Misinformation Detection            │
│  Detect fake news using multimodal AI   │
├─────────────────────────────────────────┤
│                                         │
│  📝 Text Input Area                     │
│  [Paste your text here...]              │
│                                         │
│  [🔍 Analyze Button]                    │
│                                         │
│  Result:                                │
│  ✅ REAL NEWS (87% confidence)          │
│                                         │
└─────────────────────────────────────────┘
```

---

## Deployment Options

### Option 1: Local Development (Free) ⭐ EASIEST
```bash
streamlit run app.py
```

### Option 2: Docker (Recommended for Production)
```bash
docker build -t misinformation-detector .
docker run -p 8501:8501 misinformation-detector
```

Or with docker-compose:
```bash
docker-compose up
```

### Option 3: Streamlit Cloud (Free Tier Available)
1. Push repo to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" → Select repo → Deploy!

### Option 4: Cloud Hosting
- **AWS EC2**: Run on t3.medium or better
- **Google Cloud Run**: Containerized deployment
- **Azure App Service**: Enterprise deployment
- **Heroku**: Simple one-click deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## File Structure

```
.
├── app.py                          ← Main Streamlit app (RUN THIS)
├── requirements_streamlit.txt      ← Dependencies with Streamlit
├── Dockerfile                      ← Docker image
├── docker-compose.yml              ← Docker Compose setup
├── .streamlit/config.toml          ← Streamlit configuration
│
├── checkpoints/
│   └── final_model.pt              ← Your trained model
│
├── models/                         ← Model architecture
├── data/                           ← Data preprocessing
├── inference/                      ← Prediction pipeline
└── utils/                          ← Utilities
```

---

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended (for GPU)
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA 11.8+
- 5GB disk space

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Module not found"** | Run `pip install -r requirements_streamlit.txt` |
| **"Port 8501 in use"** | Use `streamlit run app.py --server.port=8502` |
| **"Model not found"** | Ensure `checkpoints/final_model.pt` exists |
| **"GPU not detected"** | Install CUDA 11.8+ and `torch[cuda]` |
| **App is slow** | Use GPU or upgrade instance |

---

## Environment Variables (Optional)

Create `.env` file:
```
CUDA_VISIBLE_DEVICES=0          # Use specific GPU
STREAMLIT_SERVER_PORT=8501      # Custom port
STREAMLIT_LOGGER_LEVEL=info     # Log level
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Startup Time | ~3-5 seconds (cached) |
| Inference Time (CPU) | ~2-3 seconds |
| Inference Time (GPU) | ~0.5-1 second |
| Memory Usage | ~2-3 GB (model + cache) |

---

## Security Notes

✅ **Local Processing**: All computation happens on your server  
✅ **No API Keys**: No external dependencies  
✅ **Privacy**: User data never leaves your system  

For public deployment, consider:
- Using HTTPS
- Adding authentication
- Rate limiting
- Input validation

---

## Next Steps

1. **Test Locally**: `streamlit run app.py`
2. **Try Examples**: Use provided example posts
3. **Upload Images**: Test multimodal capabilities
4. **Deploy**: Choose a deployment option above

---

## Support & Documentation

- **Streamlit Docs**: https://docs.streamlit.io
- **Model Details**: See [README.md](README.md)
- **Deployment Details**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Happy deploying! 🎉**
