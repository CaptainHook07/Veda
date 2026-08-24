import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from data_preprocessing import load_data, clean_data
from feature_engineering import engineer_features

def main():
    # 1. Load and clean data
    print("Loading data...")
    # Navigate correctly if running from src directory or root directory
    data_path = "../data/raw/Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/Telco-Customer-Churn.csv"
        
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    
    # 2. Preprocess
    print("Preprocessing...")
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop(['customerID', 'Churn'], axis=1)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categorical features
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # 3. Train model
    print("Training model...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # 5. Save Model and Scaler
    print("Saving model artifacts...")
    model_dir = "../models" if not os.path.exists("models") else "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "churn_rf_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "feature_columns.pkl"))
    joblib.dump(num_cols, os.path.join(model_dir, "num_cols.pkl"))
    print("Done.")

if __name__ == "__main__":
    main()
