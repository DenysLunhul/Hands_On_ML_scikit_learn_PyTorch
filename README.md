# Hands-On Machine Learning

Personal study notebooks following **"Hands-On Machine Learning with Scikit-Learn and PyTorch" (2025)**, extended with custom CNN/RNN architectures from scratch, vision & multimodal transformers, generative models (VAEs/GANs/diffusion), LLM fine-tuning (SFT/DPO), mixed precision training, and quantization.

---

## Repository Structure

```
Hands-On_ML/
├── Classic_Machine_Learning/      # Chapters 1–8: sklearn & core ML concepts
├── Neural_Networks_Deep_Learning/ # Chapters 9–18: deep learning with PyTorch & Hugging Face
```

---

## Classic Machine Learning

Based on Part I of the book. Implemented with **scikit-learn**, **pandas**, and **NumPy**.

| Chapter | Topic | Notebook |
|---------|-------|----------|
| 1 | ML Landscape — taxonomy, types of learning, key challenges | `Chapter_1/Chapter_1.ipynb` |
| 2 | End-to-End ML Project — California Housing, pipelines, model selection | `Chapter_2/Chapter_2.ipynb` |
| 3 | Classification — MNIST, confusion matrix, precision/recall, ROC, multiclass | `Chapter_3/Chapter_3.ipynb` |
| 4 | Training Linear Models — gradient descent, regularization, logistic regression | `Chapter_4/Chapter_4.ipynb` |
| 5 | Decision Trees — CART, Gini, feature importance | `Chapter_5/Chapter_5.ipynb` |
| 6 | Ensemble Learning & Random Forests — bagging, boosting, stacking | `Chapter_6/Chapter_6.ipynb` |
| 7 | Dimensionality Reduction — PCA, t-SNE, LLE | `Chapter_7/Chapter_7.ipynb` |
| 8 | Unsupervised Learning — K-Means, DBSCAN, Gaussian Mixtures | `Chapter_8/Chapter_8.ipynb` |

### Practice Projects

| Project | Description |
|---------|-------------|
| `practice/Titanic.ipynb` | Titanic survival prediction — full pipeline from EDA to submission |
| `practice/Cars_Price_Prediction.ipynb` | Car price regression on a real-world dataset |
| `Chapter_2/rebuild_from_scratch.ipynb` | End-to-end California Housing pipeline rebuilt from scratch, unguided |

---

## Neural Networks & Deep Learning

Based on Part II of the book. Implemented with **PyTorch**, **torchvision**, and **Hugging Face** libraries.

### Book Chapters

| Chapter | Topic | Notebook |
|---------|-------|----------|
| 9 | Artificial Neural Networks — perceptrons, MLP, backprop from scratch | `Chapter_9/Chapter_9.ipynb` |
| 10 | Neural Nets with PyTorch — `nn.Module`, training loops, callbacks | `Chapter_10/Chapter_10.ipynb` |
| 11 | Training Deep Neural Networks — batch norm, dropout, optimizers, learning rate schedules | `Chapter_11/Chapter_11.ipynb` |
| 12 | Deep Computer Vision with CNNs — conv layers, pooling, object detection, localization | `Chapter_12/Chapter_12.ipynb` |
| 13 | Sequences & Time Series (RNNs) — forecasting, deep RNNs, seq2seq, WaveNet | `Chapter_13/Chapter_13.ipynb` |
| 14 | NLP with RNNs & Attention — sentiment analysis, bidirectional RNNs, Hugging Face tokenizers/pretrained embeddings/pipelines/Trainer API | `Chapter_14/Chapter_14.ipynb` |
| 15 | Transformers for NLP & Chatbots — attention, positional encodings, multihead attention, transformer architecture, NMT, BERT (encoder-only), sentence embeddings, GPT-2 generation/QA, turning an LLM into a chatbot with SFT + DPO fine-tuning via **TRL** | `Chapter_15/Chapter_15.ipynb` |
| 16 | Vision & Multimodal Transformers — Vision Transformer (ViT) from scratch, fine-tuning `ViTForImageClassification` on Oxford-IIIT Pets, zero-shot classification & image-text similarity with **CLIP** | `Chapter_16/Chapter_16.ipynb` |
| 18 | Autoencoders, GANs & Diffusion Models — stacked/tied/convolutional/sparse autoencoders, variational & discrete VAEs, GAN training loop, DDPM/DDIM diffusion from scratch, Stable Diffusion text-to-image via **diffusers** | `Chapter_18/Chapter_18.ipynb` |

### Appendix

| Topic | Notebook |
|-------|----------|
| Support Vector Machines — linear/polynomial/RBF kernel SVM classification, SVM regression | `Appendix/SVM.ipynb` |
| Mixed Precision & Quantization — FP16 training, `GradScaler`, dynamic/static/QAT quantization, 4-bit (NF4) quantized LLM inference with `bitsandbytes` | `Appendix/Mixed_Precision_Quantization.ipynb` |

### CNN Architectures (from scratch in PyTorch)

| Architecture | Notebook |
|--------------|----------|
| LeNet-5 | `CNN_Architectures/LeNet-5.ipynb` |
| AlexNet | `CNN_Architectures/AlexNet.ipynb` |
| GoogLeNet (Inception) | `CNN_Architectures/GoogLeNet.ipynb` |
| ResNet-34 | `CNN_Architectures/ResNet.ipynb` |
| SENet | `CNN_Architectures/SENet.ipynb` |
| Xception | `CNN_Architectures/Xception.ipynb` |
| WaveNet | `CNN_Architectures/WaveNet.ipynb` |

### RNN Architectures (from scratch in PyTorch)

| Architecture | Notebook |
|--------------|----------|
| Vanilla RNN | `RNN_Architectures/Default_RNN.ipynb` |
| LSTM | `RNN_Architectures/LSTM.ipynb` |
| GRU | `RNN_Architectures/GRU.ipynb` |

### Practice Projects

| Project | Description |
|---------|-------------|
| `CIFAR10/cifar10.ipynb` | Image classification on CIFAR-10, with Optuna hyperparameter tuning |
| `EMNIST/emnist.ipynb` | Handwritten character recognition on EMNIST, with Optuna hyperparameter tuning |
| `HYMENOPTERA/hymenoptera.ipynb` | Transfer learning & fine-tuning ResNet-50 on bees vs. ants |

### Char-RNN

A from-scratch character-level RNN (`Char-RNN/Char-RNN_from_scratch.ipynb`) trained on Shakespeare's text to generate new text one character at a time.

### Micrograd

A minimal autograd engine (`micrograd/engine.py` + `micrograd/nn.py`) built from scratch — implements reverse-mode automatic differentiation over a scalar value graph, inspired by Andrej Karpathy's micrograd.

### Shared Utilities

`utils/utils.py` — reusable PyTorch helpers used across notebooks: CIFAR-10 data loaders with normalization, He-initialized MLP builder, and a generic train/evaluate loop with early stopping and LR scheduler support.

---

## Key Topics Covered

- Supervised & unsupervised learning fundamentals
- Full ML pipelines: preprocessing, feature engineering, cross-validation
- Support vector machines: linear, kernelized (polynomial, RBF), and regression
- CNNs: conv layers, pooling, batch norm, skip connections, attention
- RNNs: time-series forecasting, seq2seq, 1D WaveNet-style convolutions
- Transfer learning and fine-tuning pretrained models
- NLP: sentiment analysis, tokenization, attention, transformers, BERT, GPT-2, chatbots
- LLM alignment: supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) with TRL
- Vision & multimodal transformers: Vision Transformer (ViT), CLIP zero-shot classification
- Generative models: autoencoders (stacked, tied, convolutional, sparse), VAEs, discrete VAEs, GANs, DDPM/DDIM diffusion, Stable Diffusion
- Mixed precision training and model quantization (dynamic, static, QAT, 4-bit NF4)
- Hyperparameter optimization with Optuna
- Building autograd from scratch

---

## Stack

- Python
- scikit-learn · pandas · NumPy · Matplotlib
- PyTorch · torchvision · torchmetrics
- Hugging Face Transformers · Tokenizers · Datasets · sentence-transformers · TRL · Diffusers
- bitsandbytes (quantization)
- Optuna (hyperparameter tuning)
- Jupyter Notebooks
