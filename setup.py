from setuptools import setup, find_packages

setup(
    name="transformers-rwkv7",
    version="0.1.0",
    description="HuggingFace-compatible RWKV-7 'Goose' (x070) model — training & PEFT/RL ready",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.41,<5",
        "safetensors>=0.4",
        "numpy",
    ],
    extras_require={
        "peft": ["peft>=0.11"],
        "trl": ["trl>=0.9"],
        "dev": ["pytest"],
    },
)
