from setuptools import setup, find_packages

setup(
    name="echo-feeling",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "Pillow>=10.0.0",
        "nltk>=3.8.1",
        "emoji>=2.8.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    author="Swathi Lekshmi SS, Romin Varghese, R Arun, Ivin Issac",
    description=(
        "Echo Feeling — Multimodal sentiment analysis engine for e-commerce "
        "review mining integrating text, emoji, and sticker modalities."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
