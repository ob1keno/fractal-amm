from setuptools import setup, find_packages

setup(
    name="fractal-amm",
    version="0.1.0",
    description="Fractal Automated Market Maker implementation",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)