from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="georef_pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A geospatial reference processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/georef_pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "GDAL>=3.0.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
        "tqdm>=4.60.0",
        "geopy>=2.2.0",
    ],
)