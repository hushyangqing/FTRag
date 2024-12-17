# Scientific Text Generation with Fine-tuned GPT-2 and RAG 🚀

> Fine-tuning DistilGPT-2 on scientific abstracts and implementing RAG for enhanced text generation

## 📋 Overview

This project demonstrates how to:
1. Fine-tune DistilGPT-2 on ML-ArXiv-Papers dataset (last two layers only)
2. Implement Retrieval-Augmented Generation (RAG) using the fine-tuned model

## 🛠️ Setup

### Prerequisites
- Python 3.9
- Anaconda/Miniconda

### Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Create and activate conda environment
conda env create -f environment.yml
conda activate tf_env

# Run fine-tuning
python3.9 FT.py

# Run RAG implementation
python3.9 RAG.py
```

## 🏗️ Project Structure

```
.
├── FT.py           # Fine-tuning script
├── RAG.py          # RAG implementation
├── environment.yml # Conda environment
└── README.md      # Documentation
```


### Fine-tuning Details
- **Base Model**: DistilGPT-2
- **Dataset**: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
- **Approach**: Last two layers fine-tuning
- **Purpose**: Scientific text generation

## 📚 Usage

### Fine-tuning Process (FT.py)
```python
# Example usage of fine-tuning script
python3.9 FT.py
```
[Screenshot](./images/train.png)

### RAG Implementation (RAG.py)
```python
# Example usage of RAG system
python3.9 RAG.py
```

## 📝 Citation

```bibtex
@dataset{CShorten_ML-ArXiv-Papers,
  author = {CShorten},
  title = {ML-ArXiv-Papers},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers}
}
```

## ⭐ Show your support

Give a ⭐️ if this project helped you!
