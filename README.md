# Human Activity Recognition with Large Language Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.8%2B-orange)
![Ollama](https://img.shields.io/badge/ollama-0.1.0-green)

## Overview

This repository implements a novel framework for human activity recognition using wearable sensor data with minimal labeled examples. Our approach combines self-supervised learning (SSL) techniques with Large Language Models (LLMs) to create an efficient activity recognition system that requires significantly less labeled data than traditional approaches.

By encoding raw sensor data through SimCLR and Temporal Feature Contrast (TFC) models, we transform complex time-series data into meaningful embeddings that cluster similar activities together. These embeddings serve as input to LLMs, enabling them to function as virtual annotators without requiring extensive labeled examples.

## Key Features

- **Self-supervised learning** for time-series representation learning
- **LLM-based annotation** of sensor data without extensive labeled examples
- **Local deployment** capabilities using Ollama for smaller LLMs
- **Embedding visualization** tools for understanding model performance
- **Support for multiple HAR datasets** (PAMAP2, UCI HAR, MotionSense, HHAR)

## Architecture

Our approach consists of two primary components:

1. **Self-Supervised Encoders**: Transform raw sensor data into an embedding space where similar activities cluster together
   - SimCLR (Time-Domain Representations)
   - TFC (Time-Frequency Contrastive Learning)

2. **LLM-based Annotation**: Uses the embeddings to classify activities based on proximity
   - Can be deployed with smaller LLMs using Ollama
   - Supports both binary and multi-class classification

## Installation

```bash
# Clone the repository
git clone https://github.com/username/har-ssl-llm.git
cd har-ssl-llm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama for local LLM deployment (optional)
curl -fsSL https://ollama.com/install.sh | sh
```

## Usage

### Training Self-Supervised Encoders

```bash
# Train SimCLR encoder
python unsupervised_training.py --model simclr --dataset pamap2 --epochs 200

# Train TFC encoder
python unsupervised_training.py --model tfc --dataset pamap2 --epochs 40
```

### Generating Embeddings

```bash
# Generate embeddings using trained models
python embedded_generation.py --model simclr --dataset pamap2 --checkpoint_path checkpoints/simclr_pamap2.pt
```

### LLM-based Annotation

```bash
# Use large API-based LLMs
python activity_recognition_llm.py --embeddings_path embeddings/simclr_pamap2.npy --model gpt4

# Use local Ollama-based LLMs
python activity_recognition_llm.py --embeddings_path embeddings/simclr_pamap2.npy --model ollama --ollama_model llama2
```

### Visualization

```bash
# Visualize embeddings using t-SNE
python embedded_generation.py --embeddings_path embeddings/simclr_pamap2.npy --method tsne
```

## Datasets

The project supports the following datasets:

- **PAMAP2**: 9 participants, 12 activities, IMUs on multiple body positions
- **UCI HAR**: 30 participants, 6 activities, smartphone on waist
- **MotionSense**: 24 participants, 6 activities, smartphone in trouser pocket
- **HHAR**: 9 participants, 6 activities, smartphones and smartwatches

## Project Structure

```
├── data/                  # Dataset storage
├── models/                # Model implementations
│   ├── simclr.py          # SimCLR implementation
│   └── tfc.py             # TFC implementation
├── utils/                 # Utility functions
│   ├── data_loader.py     # Dataset loading utilities
│   ├── augmentations.py   # Time-series augmentations
│   └── visualization.py   # Embedding visualization utilities
├── notebooks/             # Jupyter notebooks for analysis
├── main.py                # Script for data preprocessing and sampling
├── unsupervised_training.py # Script for training encoders
├── embedded_generation.py  # Script for generating embeddings
├── ssl_train.py           # Script for SSL training with simplified models
├── llm_integration.py     # Script for integrating LLMs
├── activity_recognition_llm.py # Script for LLM-based activity recognition
├── activity_recognition_llm_with_real_data.py # Script for LLM-based recognition with real data
├── save_embeddings_with_data.py # Script for saving embeddings and original data
├── finetune_llama.py      # Script for fine-tuning Llama model for activity recognition
├── preprocess_pamap2.py   # Script for preprocessing PAMAP2 dataset for LLM fine-tuning
└── requirements.txt       # Project dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
