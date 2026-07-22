# Hands-On Machine Learning

Personal study notebooks following **"Hands-On Machine Learning with Scikit-Learn and PyTorch" (2025)**, extended with custom CNN architectures, RNN/time-series models, and experiment tracking with MLflow + Optuna.

---

## Repository Structure

```
Hands-On_ML/
├── Classic_Machine_Learning/      # Chapters 1–8: sklearn & core ML concepts
├── Neural_Networks_Deep_Learning/ # Chapters 9–13: deep learning with PyTorch
├── MLFLow/                        # Experiment tracking & hyperparameter tuning
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

---

## Neural Networks & Deep Learning

Based on Part II of the book. Implemented with **PyTorch** and **torchvision**.

### Book Chapters

| Chapter | Topic | Notebook |
|---------|-------|----------|
| 9 | Artificial Neural Networks — perceptrons, MLP, backprop from scratch | `Chapter_9/Chapter_9.ipynb` |
| 10 | Neural Nets with PyTorch — `nn.Module`, training loops, callbacks | `Chapter_10/Chapter_10.ipynb` |
| 11 | Training Deep Neural Networks — batch norm, dropout, optimizers, learning rate schedules | `Chapter_11/Chapter_11.ipynb` |
| 12 | Deep Computer Vision with CNNs — conv layers, pooling, object detection, localization | `Chapter_12/Chapter_12.ipynb` |
| 13 | Sequences & Time Series (RNNs) — forecasting, deep RNNs, seq2seq, WaveNet | `Chapter_13/Chapter_13.ipynb` |
| 14 | NLP with RNNs & Attention — sentiment analysis, bidirectional RNNs, Hugging Face tokenizers/pretrained embeddings/pipelines/Trainer API | `Chapter_14/Chapter_14.ipynb` |
| 15 | Transformers for NLP & Chatbots — attention, positional encodings, multihead attention, transformer architecture, NMT, BERT (encoder-only), sentence embeddings, GPT-2 generation/QA, chatbot | `Chapter_15/Chapter_15.ipynb` |

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
| `CIFAR10/cifar10.ipynb` | Image classification on CIFAR-10 |
| `EMNIST/emnist.ipynb` | Handwritten character recognition on EMNIST |
| `HYMENOPTERA/hymenoptera.ipynb` | Transfer learning & fine-tuning ResNet-50 on bees vs. ants |

### Char-RNN

A from-scratch character-level RNN (`Char-RNN/Char-RNN_from_scratch.ipynb`) trained on Shakespeare's text to generate new text one character at a time.

### Micrograd

A minimal autograd engine (`micrograd/engine.py` + `micrograd/nn.py`) built from scratch — implements reverse-mode automatic differentiation over a scalar value graph, inspired by Andrej Karpathy's micrograd.

---

## MLflow & Hyperparameter Tuning

Located in `MLFLow/`.

| Notebook | Description |
|----------|-------------|
| `MlFlow.ipynb` | Experiment tracking with MLflow on FashionMNIST |
| `HyperParameterTuning.ipynb` | Automated hyperparameter search with **Optuna** + MLflow logging |

---

## Key Topics Covered

- Supervised & unsupervised learning fundamentals
- Full ML pipelines: preprocessing, feature engineering, cross-validation
- CNNs: conv layers, pooling, batch norm, skip connections, attention
- RNNs: time-series forecasting, seq2seq, 1D WaveNet-style convolutions
- Transfer learning and fine-tuning pretrained models
- NLP: sentiment analysis, tokenization, attention, transformers, BERT, GPT-2, chatbots
- Experiment tracking and reproducibility with MLflow
- Hyperparameter optimization with Optuna
- Building autograd from scratch

---

## Stack

- Python 
- scikit-learn · pandas · NumPy · Matplotlib
- PyTorch · torchvision · torchmetrics
- Hugging Face Transformers · Tokenizers · sentence-transformers
- MLflow · Optuna
- Jupyter Notebooks