# Building-LLM

A from-scratch implementation of a GPT-style large language model, built step by step to learn how LLMs work under the hood — no high-level frameworks.

## Contents

- **notebooks/** — Chapter-by-chapter walkthroughs
  - `chapter2.ipynb` — Tokenization and data loading
  - `chapter3.ipynb` — Self-attention, causal attention, and multi-head attention
  - `chapter4.ipynb` — GPT architecture: transformer blocks, layer normalization, GELU activation, and shortcut connections
  - `the-verdict.txt` — Sample text used for training/testing the tokenizer and dataloader

- **classes_functions/** — Reusable implementations extracted from the notebooks
  - `tokenizer.py` — Custom tokenizer
  - `gpt_dataset.py`, `dataloader.py` — Dataset and DataLoader for training
  - `self_attention.py` — Self-attention and multi-head attention mechanisms
  - `layer_norm.py` — Layer normalization
  - `gelu.py` — GELU activation function
  - `feed_forward.py` — Feed-forward network module
  - `shortcut_conections.py` — Residual/shortcut connections
  - `transformer.py` — Transformer block
  - `gpt_model.py` — Full GPT model assembly

## Goal

Progressively implement every core component of a GPT-like LLM (tokenization, attention, transformer blocks, and the full model) to build a solid understanding of the architecture before moving on to training and fine-tuning.

## Requirements

- Python 3.x
- PyTorch
- Jupyter Notebook

## Usage

Open the notebooks in order (chapter2 → chapter3 → chapter4) to follow the build progression, or import the modules directly from `classes_functions/`.
