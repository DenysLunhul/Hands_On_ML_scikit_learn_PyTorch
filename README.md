# Hands-On Machine Learning

Personal study notebooks following **"Hands-On Machine Learning with Scikit-Learn and PyTorch" (2025)** by Aurélien Géron, extended with custom CNN/RNN architectures from scratch, vision & multimodal transformers, and additional practice projects.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Notebooks overview](#notebooks-overview)
- [Key topics covered](#key-topics-covered)
- [Stack / requirements](#stack--requirements)
- [How to run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Quickstart

This repository contains a large collection of Jupyter notebooks used for studying classical machine learning and deep learning topics. Notebooks may require datasets, substantial compute (GPU for many deep learning notebooks), and typical ML libraries.

Recommended baseline environment:

- Python 3.9+ (3.10/3.11 recommended)
- Jupyter or JupyterLab
- (Optional) Conda for environment management

Basic steps:

1. Clone the repository:

   git clone https://github.com/DenysLunhul/Hands_On_ML_scikit_learn_PyTorch.git

2. Create a Python environment and install dependencies. There is no top-level requirements.txt in this repo; install the packages below or create your own requirements file:

   pip install jupyterlab notebook numpy pandas scikit-learn matplotlib seaborn torch torchvision torchmetrics transformers datasets sentence-transformers diffusers optuna bitsandbytes

(Adjust versions for your platform and GPU support — e.g., follow PyTorch install instructions at https://pytorch.org.)

3. Start JupyterLab or Jupyter Notebook and open the notebooks in the folders below.

---

## Repository structure

```
Hands_On_ML_scikit_learn_PyTorch/
├── Classic_Machine_Learning/      # Chapters 1–8: scikit-learn & core ML concepts
├── Neural_Networks_Deep_Learning/ # Chapters 9–18: deep learning with PyTorch & Hugging Face
├── Appendix/                      # Supplemental notebooks and experiments
├── CNN_Architectures/             # CNNs implemented from scratch (LeNet, AlexNet, ResNet, ...)
├── RNN_Architectures/             # RNN/LSTM/GRU implementations
├── practice/                      # Short practice projects (CIFAR10, Titanic, etc.)
├── utils/                         # Shared utilities used across notebooks
├── micrograd/                     # Tiny autograd implementation from scratch
├── .gitignore
└── README.md
```

---

## Notebooks overview

High-level mapping between book chapters and folders (not exhaustive):

- Classic Machine Learning (Part I, Chapters 1–8): scikit-learn, data preprocessing, pipelines, model selection, SVMs, clustering, dimensionality reduction.
- Neural Networks & Deep Learning (Part II, Chapters 9–18): PyTorch basics, training deep nets, CNNs, RNNs/seq2seq, attention and transformers, ViT, multimodal experiments, diffusion models.
- Appendix: mixed precision, quantization, advanced tools (Optuna tuning, bitsandbytes quantization, etc.).

Refer to the notebook filenames inside each folder for exact examples and runnable content.

---

## Key topics covered

- Supervised & unsupervised learning fundamentals
- Full ML pipelines: preprocessing, feature engineering, cross-validation
- SVMs, decision trees, ensembles, random forests
- CNNs: convolutional layers, pooling, batch norm, skip connections, custom architectures
- RNNs: time-series forecasting, seq2seq, character RNNs
- Transformers for NLP and vision (ViT, CLIP-style similarity)
- Generative models: autoencoders, VAEs, GANs, diffusion models
- Mixed precision training and model quantization
- Hyperparameter optimization with Optuna
- Minimal autograd implementation (micrograd)

---

## Stack / requirements

Major libraries used in notebooks (not an exhaustive pinned list):

- Python (>=3.9)
- scikit-learn · pandas · NumPy · Matplotlib · seaborn
- PyTorch · torchvision · torchmetrics
- Hugging Face: transformers · tokenizers · datasets · sentence-transformers · diffusers · TRL
- bitsandbytes (for quantized inference)
- Optuna
- Jupyter Notebook / JupyterLab

If you prefer reproducibility, create a requirements.txt or Conda environment file and pin versions before sharing or running experiments.

---

## How to run

- For small, CPU-friendly notebooks: install the Python packages above, run `jupyter lab` or `jupyter notebook`, and open the notebook file (.ipynb).
- For GPU-based training: install a GPU-enabled PyTorch build and other CUDA-aware packages. Use a machine with a compatible NVIDIA GPU and CUDA toolkit or a cloud runtime (Colab, Kaggle, Paperspace, etc.).
- Some notebooks expect datasets to be downloaded at runtime (e.g., CIFAR-10, Oxford Pets, custom datasets). The notebook usually contains the download/prepare steps.

Notes:
- Many notebooks are meant for study and demonstration — they may not include robust error handling or production-ready training loops.
- Consider running long experiments with caution: they can require hours of compute and substantial disk space.

---

## Contributing

Contributions, fixes, and improvements are welcome. If you find errors, have suggestions, or want to add notebooks:

- Open an issue describing the change or improvement.
- For code or notebook changes, submit a pull request with a clear description and minimal, focused changes.

Please keep notebook outputs cleared when submitting PRs if the change is code-only to reduce noise and large diffs.

---

## License

No license file is included in this repository. If you plan to reuse code from here, please check with the repository owner (see Contact) or add a LICENSE file to clarify reuse terms.

---

## Contact

Repo owner: DenysLunhul — https://github.com/DenysLunhul

If you'd like, I can:

- Add a requirements.txt with a reasonable baseline of pinned package versions.
- Create a CONTRIBUTING.md or License file (e.g., MIT) and open a PR with those changes.

