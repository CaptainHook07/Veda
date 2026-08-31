import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Attempt to import prediction functions
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from predict import predict_churn
    from data_preprocessing import load_data
    CAN_PREDICT = True
except Exception as e:
    CAN_PREDICT = False
    st.error(f"Error loading src modules: {e}")

@st.cache_data
def get_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Telco-Customer-Churn.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def main():
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("Identify high-risk customers and explore churn drivers.")
    
    df = get_data()
    if df is None:
        st.warning("Dataset not found. Please ensure data is downloaded.")
        return
        
    tabs = st.tabs(["Overview", "Churn Drivers", "Customer Risk Scoring"])
    
    with tabs[0]:
        st.header("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        
        # Calculate revenue, coercing errors to NaN and filling with 0
        total_rev = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0).sum()
        col3.metric("Total Revenue", f"${total_rev:,.2f}")
        
        # Workaround for pyarrow import error due to App Control Policy
        st.markdown(df.head(10).to_html(), unsafe_allow_html=True)
        
    with tabs[1]:
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn by Contract Type")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Contract', hue='Churn', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Churn by Monthly Charges")
            fig2, ax2 = plt.subplots()
            sns.kdeplot(data=df[df['Churn'] == 'No'], x='MonthlyCharges', label='No Churn', fill=True, ax=ax2)
            sns.kdeplot(data=df[df['Churn'] == 'Yes'], x='MonthlyCharges', label='Churn', fill=True, ax=ax2)
            ax2.legend()
            st.pyplot(fig2)
            
    with tabs[2]:
        st.header("Customer Risk Scoring")
        if CAN_PREDICT:
            st.markdown("Select a sample of customers to score their churn risk.")
            if st.button("Score Random 100 Customers"):
                sample_df = df.sample(100, random_state=np.random.randint(0, 1000))
                with st.spinner("Calculating risk scores..."):
                    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
                    try:
                        results = predict_churn(sample_df, model_dir=model_dir)
                        st.success("Scoring complete!")
                        
                        # Style the dataframe
                        def color_risk(val):
                            color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
                            return f'color: {color}'
                            
                        # Workaround for pyarrow import error
                        st.markdown(results.style.map(color_risk, subset=['Risk_Category']).to_html(), unsafe_allow_html=True)
                        
                        # Summary
                        high_risk_count = (results['Risk_Category'] == 'High').sum()
                        st.warning(f"Found {high_risk_count} High-Risk Customers requiring immediate action.")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}. Has the model been trained?")
        else:
            st.warning("Prediction modules unavailable.")

if __name__ == "__main__":
    main()
