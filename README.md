# neural-networks-and-digit-recognition
# 🧠 Neural Network Class Implementations From Scratch

This repository contains different types of neural network classes and neural networks created with them for digit recognition trained and tested on the **MNIST dataset**.
The implementations evolve from a simple fully-connected network to more advanced convolutional architectures, including a parallelized and optimized version for faster training.  

---

## 📖 Table of Contents
1. [Overview](#overview)  
2. [File Summaries](#file-summaries)  
   - [digit_recognition](#1-digit_recognition)  
   - [digit_recognition_conv](#2-digit_recognition_conv)  
   - [fast_digit_recognition_conv](#3-fast_digit_recognition_conv)  
3. [Results](#results)  
4. [Future Improvements](#future-improvements)

---

## 🔍 Overview
All implementations are built **from scratch in Python**(with numpy) without relying on high-level deep learning libraries.  
They demonstrate:
- Custom **Neural Network classes**  
- Implementations of **activation** and **cost functions**  
- Support for **fully connected** and **convolutional layers**  
- Experimentation with **optimization techniques** and **performance improvements**

The classes and activation/cost functions as well as training and test results are contained within a jupyter notebook.

---

## ⚙️ Technologies Used
- **Python** – Core programming language  
- **Jupyter Notebook** – For running experiments and visualizing results  
- **NumPy** – Efficient numerical computations and matrix operations  
- **Numba** – Performance optimization with:  
  - `njit`  
  - `nopython`  
  - `prange` (parallel execution)  

---

## 📂 File Summaries

### 1. `digit_recognition`
- **Description:**  
  Contains a basic **fully connected neural network** implementation.  
- **Features:**  
  - Custom `NeuralNetwork` class for building and training fully connected models  
  - Implementations of activation functions (e.g., sigmoid, ReLU)  
  - Implementations of cost functions (e.g., MSE, cross-entropy)  
  - Training and testing on the MNIST dataset  
- **Results:**  
  Achieved **97–98% accuracy**, which is quite strong for a simple fully connected NN.

---

### 2. `digit_recognition_conv`
- **Description:**  
  Extends the neural network to support **multiple layer types**, allowing more flexible architectures.  
- **Features:**  
  - Supports 4 layer types:  
    - **Dense (fully connected)**  
    - **Convolutional**  
    - **MaxPooling**  
    - **Flatten** (for reshaping outputs between convolutional and dense layers)  
  - Activation and cost functions implemented  
  - Trained and tested a CNN with max pooling  
- **Results:**  
  Accuracy was **lower compared to the fully connected network** in file 1.

---

### 3. `fast_digit_recognition_conv`
- **Description:**  
  A performance-optimized version of `digit_recognition_conv` using **Numba** for parallelization.  
- **Features:**  
  - Uses **`njit` parallelization** with `nopython` and `prange` for faster execution  
  - Implements training optimizations:
    - **Batch printing** (printing training results every *n* batches)
    - **Learning rate decay** (adjusts learning rate every *n* epochs)  
    - **Momentum (SGD with momentum)** (different update rule for momentum)
    - **Data contrast/gamma modifications** for training/testing
    - **Padding** for training/testing data
  - Helper cells for visualizing training/testing data and convolution/maxpool output sizes.
  - Supports convolutional architectures similar to file 2  
- **Results:**  
  The optimized CNN achieved **similar accuracy to the fully connected NN** (~97–98%) while running **much faster**.

---

## 📊 Results
- **Fully Connected NN (digit_recognition):** 97–98% accuracy  
- **CNN (digit_recognition_conv):** Lower accuracy than FC NN  
- **Optimized CNN (fast_digit_recognition_conv):** Similar accuracy to FC NN (97–98%), with significant speedup  

---

## 🚀 Future Improvements
Potential next steps:
- Implement **regularization** techniques (dropout, L2 weight decay)  
- Experiment with **different CNN architectures** (e.g., deeper layers, different kernel sizes)  
- Add support for **batch normalization**  
- Explore **GPU acceleration** (e.g., via CUDA or PyTorch for comparison)  

---
