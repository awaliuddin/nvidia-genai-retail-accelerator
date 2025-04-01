from setuptools import setup, find_packages

setup(
    name="nvidia-genai-retail-accelerator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal GenAI solution for retail applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nvidia-genai-retail-accelerator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
)
