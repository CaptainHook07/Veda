import pandas as pd
import numpy as np

def engineer_features(df):
    """Creates new predictive features."""
    df_fe = df.copy()
    
    # Tenure groups
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-1 yr', '1-2 yrs', '2-3 yrs', '3-4 yrs', '4-5 yrs', '5-6 yrs']
    df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Average monthly revenue calculated
    df_fe['CalculatedAvgMonthly'] = df_fe['TotalCharges'] / df_fe['tenure'].replace(0, 1)
    
    # Number of subscribed services
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Create a count of services
    df_fe['ServiceCount'] = 0
    for col in services:
        if col in df_fe.columns:
            # 'Yes' adds 1, anything else is 0
            df_fe['ServiceCount'] += (df_fe[col] == 'Yes').astype(int)
            
    return df_fe
