# Language Modeling with LSTM and AWD-LSTM on WikiText-2

This repository contains code for language modeling using **LSTM** and **AWD-LSTM** (ASGD Weight-Dropped LSTM) architectures, trained on the **WikiText-2** dataset. The goal of this project is to explore the performance improvements gained by applying advanced regularization and optimization techniques to a basic LSTM model, as introduced in the AWD-LSTM paper.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Base LSTM Model](#base-lstm-model)
  - [AWD-LSTM Model](#awd-lstm-model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Introduction

Language modeling is a critical task in natural language processing (NLP) that involves predicting the next word in a sequence given its preceding context. This project builds and evaluates two models for this task:

1. **Base Model**: A traditional LSTM-based language model.
2. **Improved Model**: An AWD-LSTM model that incorporates advanced regularization techniques like weight dropping, variational dropout, and optimization strategies such as ASGD to enhance performance and generalization.

## Dataset

The project uses the **WikiText-2** dataset, which contains Wikipedia articles with a vocabulary size of around 33,000 words. It is commonly used for language modeling tasks because of its rich linguistic structure and long-range dependencies. The dataset is pre-tokenized and divided into training, validation, and test sets.

- **Training set**: ~2 million tokens
- **Validation set**: ~200,000 tokens
- **Test set**: ~200,000 tokens

## Models

### Base LSTM Model

The base model is a two-layer LSTM network. LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) that can capture long-term dependencies in sequential data, making them suitable for language modeling tasks.

**Key Features**:
- Two-layer LSTM
- Fully connected layer for word prediction
- Trained using cross-entropy loss and evaluated with perplexity

### AWD-LSTM Model

The AWD-LSTM model improves upon the base LSTM model by introducing several key techniques from the 2017 paper "Regularizing and Optimizing LSTM Language Models" by Stephen Merity et al. These include:

- **Weight Dropping**: Dropout applied to the LSTM’s recurrent weights to prevent overfitting.
- **ASGD**: Averaged Stochastic Gradient Descent to smooth training and prevent overfitting.
- **Variational Dropout**: Dropout applied consistently across time steps in the LSTM, ensuring temporal consistency.
- **Weight Tying**: Sharing weights between the input embedding and output layers to reduce model size and improve generalization.

## Results

The models were evaluated using **perplexity**, a metric that measures the uncertainty of the model in predicting the next word in the sequence. Lower perplexity indicates better performance.

| **Model**        | **Validation Perplexity** | **Test Perplexity** |
|------------------|---------------------------|---------------------|
| LSTM (Base)      | 128.3                     | 121.7               |
| AWD-LSTM         | 80.27                     | 77.11               |

The AWD-LSTM model significantly outperformed the base LSTM model, demonstrating the effectiveness of the advanced regularization techniques.


