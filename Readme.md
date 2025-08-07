# Product Ranking Recommendation System

This project implements a feature-based neural ranking model to recommend products for e-commerce users. The system uses PyTorch for model training and pandas for data preparation and manipulation.

---

## Overview

Given raw event and transaction data, this pipeline prepares features representing user-product interactions and trains a neural network to predict how likely a user is to buy (or re-buy) products. The main goal is to personalize product rankings for users.

---

## Components

### 1. Data Loading (`load_data.py`)
- **`load_events()`**: Loads event data from a CSV file.
- **`filter_transactions()`**: Filters for transaction events, yielding ordered purchase records for each user.

### 2. Feature Engineering (`features.py`)
- **User-item features generated:**
  - **`buy_count`**: How many times a user has bought a product.
  - **`total_user_txns`**: Total purchases made by a user.
  - **`global_product_count`**: Popularity of each product in the dataset.
  - **`recency_days`**: How recently the user bought that product.
  - **`sequence_score`**: How often this product is bought directly after another in the same user’s sequence.

### 3. Dataset Preparation (`dataset.py`)
- **`ProductRankingDataset`**: PyTorch dataset that yields a dictionary per sample with:
  - Encoded user and product indices
  - Engineered features (recency, sequence, total transactions)
  - Regression label (`buy_count`)

### 4. Model Architecture (`model.py`)
- **`ProductRankingModel`**: PyTorch neural model with:
  - Embeddings for users and items
  - Feedforward MLP combining embeddings and features
  - Outputs a ranking score or prediction for the user-product pair

---

## Workflow

1. **Load events data** and filter for transactions only.
2. **Compute and merge user-product features** to form the modeling dataset.
3. **Index users and items** for embedding inputs.
4. **Prepare PyTorch dataset**.
5. **Train ranking model** using user, item, and engineered features.

---

## Intended Usage

- Generates personalized product rankings for users, utilizing behavioral and temporal features.
- Provides a base for enhancements including additional features, sampling strategies, or model architectures.

---

## Example Features

- User index, item index
- `recency_days`, `sequence_score`, `total_user_txns`
- Target: `buy_count`

---

## Folder Structure

