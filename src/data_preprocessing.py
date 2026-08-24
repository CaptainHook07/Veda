import pandas as pd
import numpy as np

def load_data(filepath="data/raw/Telco-Customer-Churn.csv"):
    """Loads the dataset."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Cleans the dataset: handles missing values, fixes data types."""
    df_clean = df.copy()
    
    # 'TotalCharges' is object because of empty strings ' ' for new customers
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])
    
    # Fill missing TotalCharges with 0 (since they are new customers, tenure=0)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    return df_clean
