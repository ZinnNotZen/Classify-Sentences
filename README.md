# Sentiment Analysis Using Neural Networks and NLP

## Overview

This project aims to develop a neural network model that classifies customer comments as either **positive** or **negative** based on sentiment analysis. The analysis leverages natural language processing (NLP) techniques on the **UCI Sentiment Labeled Sentences Dataset**, enabling organizations to extract actionable insights from customer feedback for improved decision-making.

---

## Research Question

**How accurately can a neural network model classify sentences as positive or negative based on sentiment analysis of customer comments?**

This question addresses the real-world organizational need to automate sentiment classification of customer feedback, allowing businesses to quickly gauge customer satisfaction and respond appropriately.

---

## Objectives and Goals

- **Objective:** Develop a program that accurately predicts the sentiment (positive or negative) of a given customer comment.
- **Goal:** Train and evaluate a neural network using NLP techniques to achieve above **85% accuracy** on the sentiment classification task, creating a model suitable for real-world deployment in organizational settings.

---

## Neural Network Architecture

### Model Type

- **Recurrent Neural Network (RNN):**  
  Selected for its ability to process sequential data and capture contextual dependencies between words, which is essential for accurate sentiment classification.

### Network Layers

1. **Embedding Layer:**  
   Converts words into 50-dimensional dense vectors; vocabulary size is 2,816 unique words (total parameters: 140,800).  
   Output shape: `[batch_size, 24 (max sequence length), 50]`.

2. **Fully Connected Layer 1:**  
   Transforms embeddings to 64-dimensional hidden vectors (3,264 parameters).  
   Uses **ReLU** activation for non-linearity.

3. **Fully Connected Layer 2 (Output Layer):**  
   Outputs a single value representing sentiment probability with a **sigmoid** activation function (65 parameters).  
   Output shape: `[batch_size, 1]`.

### Total Trainable Parameters: 144,129

---

## Data Preparation

### Exploratory Data Analysis

- **Unusual Characters:** 121 sentences contained characters like `+` or typos (e.g., "seperated"). Minimal cleaning performed to improve model robustness.
- **Vocabulary Size:** 2,816 unique tokens after tokenization.
- **Word Embedding Length:** 50 dimensions chosen for a balance between representational capacity and training efficiency.
- **Maximum Sequence Length:** 25 words (captures 95% of sentences; average length = 11.25 words).

### Tokenization and Normalization

- Text normalized by converting to lowercase and removing non-alphabetic characters.
- Tokenization performed via simple text splitting.
- Used Python's built-in `re` library for cleaning text.

### Sequence Padding

- Sequences padded **before** the text with zeros to the maximum length of 25.
- This preserves recent and important tokens towards the end of sequences.

### Sentiment Categories and Activation

- **Two categories:** Positive (1) and Negative (0).
- **Activation function:** Sigmoid in the final layer for probability output.

### Dataset Splitting

- Training: 80% of data
- Testing: 20% of data
- No separate validation set due to dataset size; cross-validation or tuning may be added later.

---

## Model Training and Evaluation

### Hyperparameters

- **Activation Functions:** ReLU (hidden layer), Sigmoid (output layer)
- **Loss Function:** Binary Cross-Entropy (BCELoss)
- **Optimizer:** Adam optimizer for adaptive learning rate
- **Epochs:** 20 epochs to balance learning and avoid overfitting
- **Batch Size:** 5 for efficient memory use

### Training Process

- Early stopping was considered to prevent overfitting but a fixed 20-epoch training was chosen for consistency.
- Training and validation loss monitored to detect underfitting or overfitting.

### Results

- Training and validation accuracy both remained high and close, demonstrating good generalization.
- Final accuracy surpassed the goal threshold of 85%.

---

## Ethical Considerations

- Data preprocessing aimed to minimize bias, including uniform encoding and normalization.
- Model transparency ensured by documenting architecture and training methodology.
- Accountability maintained by evaluating performance metrics and monitoring for overfitting.
- The model avoids favoring any specific demographic group by focusing solely on text sentiment.

---

## Files Pro
