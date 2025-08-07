# src/load_data.py

import pandas as pd

def load_events(filepath='data/events.csv'):
    df = pd.read_csv(filepath)
    print("Loaded events.csv with shape:", df.shape)
    return df

def filter_transactions(events_df):
    df_txn = events_df[events_df['event'] == 'transaction']
    df_txn = df_txn[['visitorid', 'itemid', 'timestamp']]
    df_txn = df_txn.sort_values(by='timestamp')
    return df_txn
