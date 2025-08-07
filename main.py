# main.py

# %%
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.load_data import load_events, filter_transactions
from src.features import compute_user_item_features
from src.dataset import ProductRankingDataset
from src.model import ProductRankingModel


# %%

# Load and preprocess
events = load_events()
transactions = filter_transactions(events)

# Feature Engineering
features_df = compute_user_item_features(transactions)
print("dataset shape:", features_df.shape)
features_df = features_df.dropna()

# Drop duplicates (same user-item pair)
features_df = features_df.drop_duplicates(subset=['visitorid', 'itemid'])

print("Cleaned dataset shape:", features_df.shape)

print(features_df.head())

# %%

# --- Step 4: Encode visitorid and itemid into indices ---

# Create mappings
unique_users = features_df['visitorid'].unique()
unique_items = features_df['itemid'].unique()

user2index = {user_id: idx for idx, user_id in enumerate(unique_users)}
item2index = {item_id: idx for idx, item_id in enumerate(unique_items)}

# Map IDs to indices
features_df['user_index'] = features_df['visitorid'].map(user2index)
features_df['item_index'] = features_df['itemid'].map(item2index)

print("Total unique users:", len(user2index))
print("Total unique items:", len(item2index))
print(features_df[['visitorid', 'user_index', 'itemid', 'item_index']].head())

train_df, val_df = train_test_split(features_df, test_size=0.2, random_state=42)

# --- Create PyTorch Dataset & Dataloader ---
train_dataset = ProductRankingDataset(train_df)
val_dataset = ProductRankingDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# %%

dataset = ProductRankingDataset(features_df)

# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

# -------------------------------
# Step 8: Initialize Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProductRankingModel(num_users=len(user2index), num_items=len(item2index)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%



# -------------------------------
# Step 9: Train Loop
# -------------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        feats = batch['features'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(user, item, feats)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            feats = batch['features'].to(device)
            label = batch['label'].to(device)

            outputs = model(user, item, feats)
            loss = criterion(outputs, label)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader):.4f}")

# %%
# Get top 500 most interacted items
top_items = features_df['item_index'].value_counts().head(500).index.tolist()

# Create fewer user-item pairs: 11k × 500 = 5.8 million (safe)
all_user_indices = features_df['user_index'].unique()

user_item_pairs = pd.DataFrame(
    [(u, i) for u in all_user_indices for i in top_items],
    columns=['user_index', 'item_index']
)

# %%
# Merge with feature dataframe to get features
predict_df = user_item_pairs.merge(
    features_df,
    on=['user_index', 'item_index'],
    how='left'
)

# Fill missing features with mean or 0
for col in ['recency_days', 'sequence_score', 'total_user_txns']:
    predict_df[col] = predict_df[col].fillna(features_df[col].mean())

predict_dataset = ProductRankingDataset(predict_df)
predict_loader = DataLoader(predict_dataset, batch_size=1024, shuffle=False)

# %%
model.eval()
all_scores = []
all_user_ids = []
all_item_ids = []

with torch.no_grad():
    for batch in predict_loader:
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        feats = batch['features'].to(device)
        scores = model(user, item, feats)

        all_scores.extend(scores.cpu().numpy())
        all_user_ids.extend(batch['user'].cpu().numpy())
        all_item_ids.extend(batch['item'].cpu().numpy())

# %%
scores_df = pd.DataFrame({
    'user_index': all_user_ids,
    'item_index': all_item_ids,
    'score': all_scores
})

# Get top 20 items per user
top_n = 20
top_recommendations = scores_df.sort_values(by=['user_index', 'score'], ascending=[True, False])
top_recommendations = top_recommendations.groupby('user_index').head(top_n)

print(top_recommendations.head(10))

# %%

idx2user = {idx: user_id for user_id, idx in user2index.items()}
idx2item = {idx: item_id for item_id, idx in item2index.items()}

# Map back to original IDs
top_recommendations['user_id'] = top_recommendations['user_index'].map(idx2user)
top_recommendations['product_id'] = top_recommendations['item_index'].map(idx2item)

# Rename score column for clarity
top_recommendations = top_recommendations.rename(columns={'score': 'predicted_score'})

# Save
top_recommendations[['user_id', 'product_id', 'predicted_score']].to_csv('top5_recommendations.csv', index=False)
# %%
