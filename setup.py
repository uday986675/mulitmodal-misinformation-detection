"""
Setup configuration for Multimodal Misinformation Detection project.
"""

from setuptools import setup, find_packages

setup(
    name="multimodal-misinformation-detection",
    version="1.0.0",
    description="Deep learning system for detecting misinformation in multimodal social media content",
    author="Uday Kiran",
    author_email="sambhanaudaykiran@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core ML libraries
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        
        # Web framework
        "streamlit>=1.28.0",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "opencv-python-headless>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=training.train:main",
            "predict=inference.predict:main",
        ],
    },
)
