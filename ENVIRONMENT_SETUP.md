# Environment Configuration Examples

## .env File (Optional)

Create a `.env` file in the project root to customize environment variables:

```bash
# .env file
CUDA_VISIBLE_DEVICES=0
STREAMLIT_SERVER_PORT=8501
STREAMLIT_LOGGER_LEVEL=info
PYTHONUNBUFFERED=1
```

Then load with:
```bash
source .env  # Linux/Mac
# or Windows users can set in system environment
```

---

## Python Virtual Environment Setup

### Linux/Mac
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements_streamlit.txt

# Run app
streamlit run app.py

# Deactivate when done
deactivate
```

### Windows
```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements_streamlit.txt

# Run app
streamlit run app.py

# Deactivate when done
deactivate
```

---

## Conda Environment Setup

If you prefer Conda:

```bash
# Create conda environment
conda create -n misinformation-detector python=3.10

# Activate
conda activate misinformation-detector

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirements_streamlit.txt

# Run app
streamlit run app.py
```

---

## GPU Setup (Optional)

### NVIDIA GPU with CUDA

```bash
# Check CUDA installation
nvcc --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

### AMD GPU (ROCm)

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Docker Environment

### Dockerfile Environment Variables

```dockerfile
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
```

### Docker Run with Environment

```bash
docker run -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  misinformation-detector
```

### Docker Compose Environment

```yaml
services:
  app:
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_PORT=8501
```

---

## Streamlit Configuration (`.streamlit/config.toml`)

### Basic Configuration
```toml
[server]
port = 8501
headless = true
runOnSave = true

[logger]
level = "info"

[client]
showErrorDetails = true
```

### Production Configuration
```toml
[server]
port = 8501
headless = true
runOnSave = false
maxUploadSize = 200
enableXsrfProtection = true

[logger]
level = "warning"

[client]
showErrorDetails = false
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### High-Traffic Configuration
```toml
[server]
port = 8501
headless = true
runOnSave = false
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
enableWSMessageCompression = true

[logger]
level = "warning"

[client]
toolbarMode = "minimal"
showErrorDetails = false

[theme]
primaryColor = "#1f77b4"
```

---

## Production Environment Variables

### AWS EC2
```bash
# Set in /etc/environment or .bashrc
export CUDA_VISIBLE_DEVICES=0
export STREAMLIT_SERVER_PORT=80
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export PYTHONUNBUFFERED=1
export MODEL_PATH=/app/checkpoints/final_model.pt
```

### Google Cloud Run
In `app.yaml`:
```yaml
env:
  - name: STREAMLIT_SERVER_PORT
    value: "8080"
  - name: STREAMLIT_SERVER_ADDRESS
    value: "0.0.0.0"
  - name: PYTHONUNBUFFERED
    value: "1"
```

### Heroku
In `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Logging Configuration

### Development (Verbose)
```toml
[logger]
level = "debug"
```

### Testing (Standard)
```toml
[logger]
level = "info"
```

### Production (Quiet)
```toml
[logger]
level = "warning"
```

---

## Common Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` or `0,1` |
| `STREAMLIT_SERVER_PORT` | Port number | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | `0.0.0.0` |
| `PYTHONUNBUFFERED` | Output buffering | `1` |
| `MODEL_PATH` | Model location | `/app/checkpoints/final_model.pt` |
| `TORCH_HOME` | PyTorch cache | `/app/.torch` |

---

## Performance Tuning

### For CPU-Only Systems
```bash
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
streamlit run app.py
```

### For GPU Systems
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
streamlit run app.py
```

### For Memory-Constrained Systems
```bash
export PYTHONHASHSEED=0
streamlit run app.py \
  --logger.level=warning \
  --client.showErrorDetails=false
```

---

## Health Check Configuration

### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
```

### Monitoring Script
```bash
#!/bin/bash
while true; do
    if curl -f http://localhost:8501/_stcore/health; then
        echo "App is healthy"
    else
        echo "App is down!"
        # Send alert or restart
    fi
    sleep 30
done
```

---

## Troubleshooting Environment Issues

### Python Path Issues
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
python app.py
```

### Module Import Errors
```bash
pip install -r requirements_streamlit.txt --upgrade
```

### CUDA Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### Memory Issues
```bash
# Check available memory
free -h

# Limit Python memory
ulimit -v 4000000

# Or in Docker
docker run -m 4g misinformation-detector
```

---

## Quick Start with Environment

### Complete Setup Script
```bash
#!/bin/bash
set -e

echo "🚀 Setting up Misinformation Detector"

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements_streamlit.txt

# Verify
python test_setup.py

# Run
echo "✅ Setup complete! Running app..."
streamlit run app.py
```

### Windows Setup Script
```cmd
@echo off
echo 🚀 Setting up Misinformation Detector

python -m venv venv
call venv\Scripts\activate

pip install -r requirements_streamlit.txt

python test_setup.py

echo ✅ Setup complete! Running app...
python -m streamlit run app.py
```

---

**Choose your environment setup and get started!** 🎉
