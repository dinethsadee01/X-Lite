"""
Setup script for X-Lite package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="x-lite",
    version="0.1.0",
    author="Dineth Sadee",
    description="Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dinethsadee01/X-Lite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
)
