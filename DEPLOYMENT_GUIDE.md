# Streamlit Deployment Guide

## Quick Start - Local Development

### 1. Install Streamlit
```bash
pip install streamlit==1.28.0
```

Or update your requirements file:
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the App Locally
```bash
cd "Multimodal Misinformation Detection in Noisy Social Streams"
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## Features

✅ **Text-only Analysis**: Detect misinformation from text alone  
✅ **Multimodal Analysis**: Combine text + image for better accuracy  
✅ **Real-time Predictions**: Instant confidence scores and probabilities  
✅ **User-Friendly Interface**: Clean, intuitive design  
✅ **Privacy-First**: All processing happens locally  
✅ **GPU Support**: Automatic CUDA detection for faster inference  

---

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)
1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repo
4. Select the `app.py` file
5. Deploy in one click!

**Cost**: Free tier available

---

### Option 2: Docker Container

#### Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
docker build -t misinformation-detector .
docker run -p 8501:8501 misinformation-detector
```

Access at: **http://localhost:8501**

---

### Option 3: AWS EC2 / Google Cloud / Azure

1. **Launch an instance** (t3.medium or larger recommended for GPU)
2. **SSH into the instance**
3. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_streamlit.txt
   ```
4. **Run the app**:
   ```bash
   streamlit run app.py --server.port=80
   ```
5. **Make it persistent** using `screen` or `nohup`:
   ```bash
   nohup streamlit run app.py > app.log 2>&1 &
   ```

---

### Option 4: Heroku

#### Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Create `runtime.txt`:
```
python-3.10.0
```

#### Deploy:
```bash
heroku create your-app-name
git push heroku main
heroku logs --tail
```

---

## Configuration for Production

### Streamlit Config File (`.streamlit/config.toml`)
```toml
[server]
port = 8501
headless = true
runOnSave = true
maxUploadSize = 200

[logger]
level = "info"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

---

## Performance Tips

1. **GPU Support**: Uses CUDA automatically if available
2. **Model Caching**: Uses Streamlit's `@st.cache_resource` for fast startup
3. **Batch Processing**: Can handle single or batch predictions
4. **Memory Optimization**: Model is loaded once and reused

---

## Troubleshooting

### Model Not Found
- Ensure checkpoint exists at `checkpoints/final_model.pt`
- Or update path in `app.py` line ~39

### Out of Memory
- Use smaller batch size
- Run on GPU if available
- Restart Streamlit session

### Image Processing Issues
- Ensure PIL/Pillow is installed
- Check image format (JPG, PNG, BMP, GIF supported)
- Resize large images before upload

### Port Already in Use
```bash
streamlit run app.py --server.port=8502
```

---

## API Alternative

For programmatic access, you can also use the Predictor class directly:

```python
from inference.predict import Predictor
from data.preprocess_text import TextPreprocessor
from models.classifier import CompleteMultimodalModel

# Load model
model = CompleteMultimodalModel(config)
model.load_state_dict(torch.load('checkpoints/final_model.pt'))

# Create predictor
predictor = Predictor(model, text_preprocessor, image_preprocessor)

# Make prediction
result = predictor.predict_single("Your text here", "path/to/image.jpg")
print(result['predictions'], result['confidence'])
```

---

## Support

For issues or questions:
1. Check model checkpoint exists
2. Verify all dependencies installed
3. Review log files in `logs/` directory
4. Check Streamlit documentation: https://docs.streamlit.io

---

**Deployed with ❤️ using Streamlit**
