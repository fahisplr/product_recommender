# src/features.py

import pandas as pd

def compute_user_item_features(df_txn):
    # ---------------------------
    # buy_count: how many times a user bought a product
    # ---------------------------
    df = df_txn.groupby(['visitorid', 'itemid']).size().reset_index(name='buy_count')

    # ---------------------------
    # total_user_txns: total number of purchases by a user
    # ---------------------------
    total_user_txns = df.groupby('visitorid')['buy_count'].sum().reset_index(name='total_user_txns')
    df = df.merge(total_user_txns, on='visitorid')

    # ---------------------------
    # global_product_count: total number of times each product is bought (popularity)
    # ---------------------------
    product_popularity = df_txn.groupby('itemid').size().reset_index(name='global_product_count')
    df = df.merge(product_popularity, on='itemid')

    # ---------------------------
    # recency_days: how recently the user bought this product
    # ---------------------------
    df_txn['timestamp'] = pd.to_datetime(df_txn['timestamp'], unit='ms')
    last_time = df_txn['timestamp'].max()

    latest_interactions = df_txn.groupby(['visitorid', 'itemid'])['timestamp'].max().reset_index()
    latest_interactions['recency_days'] = (last_time - latest_interactions['timestamp']).dt.days

    df = df.merge(latest_interactions[['visitorid', 'itemid', 'recency_days']], on=['visitorid', 'itemid'])
    seq_score_df = compute_sequential_score(df_txn)
    
    df = df.merge(seq_score_df, on='itemid', how='left')
    df['sequence_score'] = df['sequence_score'].fillna(0)

    return df

def compute_sequential_score(df_txn):
    """
    Returns a DataFrame of (itemid) → score based on how often it appears
    right after another product in the same user's purchase sequence.
    """
    df_txn = df_txn.copy()
    df_txn['timestamp'] = pd.to_datetime(df_txn['timestamp'], unit='ms')
    df_txn = df_txn.sort_values(by=['visitorid', 'timestamp'])

    transitions = []
    
    for user_id, group in df_txn.groupby('visitorid'):
        items = list(group['itemid'])
        for i in range(len(items) - 1):
            transitions.append((items[i], items[i+1]))

    transition_df = pd.DataFrame(transitions, columns=['prev_item', 'next_item'])
    
    # Count how often each item appears as a next-item
    next_counts = transition_df.groupby('next_item').size().reset_index(name='sequence_score')
    next_counts = next_counts.rename(columns={'next_item': 'itemid'})
    
    return next_counts

