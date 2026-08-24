import pandas as pd
import numpy as np
import joblib
import os

from data_preprocessing import clean_data
from feature_engineering import engineer_features

def load_artifacts(model_dir="models"):
    model = joblib.load(os.path.join(model_dir, "churn_rf_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    num_cols = joblib.load(os.path.join(model_dir, "num_cols.pkl"))
    return model, scaler, feature_cols, num_cols

def predict_churn(df_raw, model_dir="models"):
    """
    Given a raw dataframe of customers, returns predictions and risk categories.
    """
    model, scaler, feature_cols, num_cols = load_artifacts(model_dir)
    
    customer_ids = df_raw['customerID']
    
    df = clean_data(df_raw)
    df = engineer_features(df)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'customerID' in cat_cols: cat_cols.remove('customerID')
    if 'Churn' in cat_cols: cat_cols.remove('Churn')
    
    df_processed = pd.get_dummies(df.drop(['customerID'], axis=1, errors='ignore'), columns=cat_cols, drop_first=True)
    
    # Align columns
    for col in feature_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    df_processed = df_processed[feature_cols]
    
    # Scale numerical features
    try:
        df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    except Exception as e:
        print("Warning: scaler could not transform.", e)

    # Predict
    probas = model.predict_proba(df_processed)[:, 1]
    
    # Risk categories
    risk_categories = []
    actions = []
    for p in probas:
        if p > 0.7:
            risk_categories.append("High")
            actions.append("Immediate outreach, offer retention discount.")
        elif p > 0.4:
            risk_categories.append("Medium")
            actions.append("Send personalized engagement email.")
        else:
            risk_categories.append("Low")
            actions.append("Standard monitoring.")
            
    results = pd.DataFrame({
        'CustomerID': customer_ids,
        'Churn_Probability': probas,
        'Risk_Category': risk_categories,
        'Recommended_Action': actions
    })
    
    return results
