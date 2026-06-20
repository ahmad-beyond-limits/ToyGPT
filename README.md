## Project Overview

This repository contains the architecture, methodology, and implementation details for a character-level Generative Pre-trained Transformer built entirely from scratch. The primary objective of this project is to construct a functional autoregressive language model capable of predicting and generating coherent text based on the Karpathy Tiny Shakespeare dataset.

---

## Methodology

The system is constructed incrementally. It begins with a foundational bigram model and systematically introduces advanced neural network components to scale up to a complete multi-layer transformer architecture.

### 1. Data Acquisition and Preprocessing

* **Dataset:** The model is trained on a text corpus consisting of approximately 1.11 million characters.
* **Tokenization:** A custom character-level tokenizer is implemented. The dataset contains 65 unique characters, which are mapped to distinct integer indices for computational processing.
* **Data Split:** The tokenized corpus is divided into a 90% training partition and a 10% validation partition to measure generalization.
* **Batching Strategy:** Data is processed in parallel batches to optimize computational efficiency. Each training sequence is constrained by a defined block size, serving as the context window to predict the immediate subsequent character.

### 2. Core Architectural Components

The model architecture replicates the fundamental components of the transformer decoder block.

* **Embeddings:** The input integer sequences are transformed into dense vectors using a token embedding table. Positional embeddings are subsequently added to provide the network with relative spatial awareness of the token sequence.
* **Self-Attention Mechanism:** The model computes relationship scores between characters using linear projections of Query (Q), Key (K), and Value (V) matrices. A lower-triangular mask is applied to the attention scores to ensure the model only attends to preceding characters, preserving the autoregressive property.
* **Multi-Head Attention:** Multiple self-attention heads process the input sequence simultaneously. Their outputs are concatenated and linearly projected to capture diverse representational subspaces.
* **Feed-Forward Network:** A two-layer fully connected network utilizing a ReLU activation function processes the outputs from the attention mechanism.
* **Transformer Block:** The multi-head attention and feed-forward networks are encapsulated within sequential layers. Each block integrates layer normalization and residual connections to stabilize gradients and improve deep network training.

### 3. Mathematical Foundation

The scaled dot-product attention mechanism utilized in this architecture is calculated using the standard formulation:

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

Where $d_k$ represents the dimensionality of the key vectors. The objective function utilized for parameter optimization across the entire network is Cross-Entropy Loss.

---

## Experimental Scaling and Hyperparameters

To improve the generation quality and capture deeper linguistic patterns, the baseline architecture was scaled up significantly. The final model configuration used for the extended training run is detailed in the table below.

| Parameter | Value | Description |
| --- | --- | --- |
| Block Size | 64 | Maximum context sequence length for predictions |
| Batch Size | 256 | Number of independent sequences processed in parallel |
| Embedding Dimension | 384 | Size of the token and positional embedding vectors |
| Attention Heads | 6 | Number of parallel attention mechanisms per block |
| Transformer Layers | 6 | Total number of sequential transformer blocks |
| Dropout Rate | 0.2 | Regularization factor applied to prevent overfitting |
| Optimizer | AdamW | Optimization algorithm with a learning rate of 0.0001 |

---

## Results and Observations

The final scaled model was trained for 100,000 iterations. Loss metrics were recorded periodically to evaluate both learning progress and model generalization.

* **Training Loss:** The training loss decreased consistently throughout the operation, reaching a final value of approximately 0.28.
* **Validation Loss:** The validation loss initially decreased but began to diverge and increase after roughly 10,000 iterations, concluding at approximately 2.50.

> Note: The distinct divergence between the training and validation loss indicates that the model began to overfit the dataset. After the 10,000 step mark, the network started memorizing the training data rather than learning generalizable linguistic rules.

Despite the observable overfitting, the qualitative output of the final scaled model improved substantially compared to the earlier, smaller iterations. The generated text successfully replicates the structural formatting, syntax, and vocabulary distribution of the Shakespearean training corpus.
