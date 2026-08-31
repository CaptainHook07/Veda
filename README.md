# Customer Churn Prediction for Subscription Businesses

## Project Overview
This project is an end-to-end data science solution designed to predict customer churn in a subscription-based business (using the IBM Telco Customer Churn dataset). The system identifies high-risk customers and provides actionable recommendations to improve customer retention.

## Business Problem
Customer acquisition is often much more expensive than retention. Subscription businesses face the constant threat of churn. By proactively identifying which customers are most likely to leave, businesses can allocate retention budgets effectively and offer targeted incentives (e.g., discounts, personalized support) to save high-value accounts.

## Dataset
- **Source**: [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Features**: Demographics, account information (tenure, contract type, payment method), services subscribed, and charges.
- **Target**: `Churn` (Yes/No)

## Methodology
1. **Data Preprocessing & Cleaning**: Handled missing values in `TotalCharges` for new customers.
2. **Exploratory Data Analysis (EDA)**: Investigated churn rates by contract type, tenure, and monthly charges. 
3. **Feature Engineering**: Created `TenureGroup`, calculated average monthly charges, and aggregated total service subscriptions (`ServiceCount`).
4. **Modeling**: Used a Random Forest Classifier as a robust baseline. Handled class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
5. **Evaluation**: Evaluated on Accuracy, ROC-AUC, and F1-Score (precision/recall for the churned class is critical).

## Key Findings (Exploratory)
- **Contract Type**: Month-to-month contracts have a significantly higher churn rate compared to 1-year or 2-year contracts.
- **Tenure**: The highest risk of churn occurs within the first few months (0-1 year).
- **Monthly Charges**: Customers with higher monthly charges are more likely to churn, indicating price sensitivity.

## Business Recommendations
1. **Incentivize Long-Term Contracts**: Offer discounts for upgrading from month-to-month to annual contracts.
2. **Proactive Outreach for New Users**: Focus retention efforts on the first 6-12 months of a customer's tenure.
3. **Targeted Discounts**: For customers predicted as 'High Risk' due to high monthly charges, offer targeted retention discounts.

## How to Run & Use the App

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Data Pipeline**:
   Download the data, clean it, and train the prediction model:
   ```bash
   python download_data.py
   python src/data_preprocessing.py
   python src/train_model.py
   ```

3. **Run Streamlit Dashboard**:
   If `streamlit run` gives you an error, use the Python module command:
   ```bash
   python -m streamlit run app/streamlit_app.py
   ```

### What Inputs to Give the App
Once the app launches in your web browser, you will see a sidebar/form where you can enter a hypothetical or real customer's details.

**Key Inputs to Test:**
- **Tenure:** Number of months the customer has been with the company. Lower tenure usually increases churn risk.
- **Contract:** Month-to-month contracts have much higher churn rates compared to One or Two year contracts.
- **Internet Service:** Fiber optic users might have different churn behaviors than DSL.
- **Monthly Charges:** Higher charges can indicate price sensitivity.

Adjust the sliders and dropdowns to represent different customer profiles, and click the **Predict** button at the bottom to see whether the AI thinks that customer will stay or cancel!

## Technologies Used
- **Python**: pandas, numpy, scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Application**: Streamlit, joblib

## Future Improvements
- Implement XGBoost or LightGBM for potentially higher predictive performance.
- Integrate SHAP (SHapley Additive exPlanations) for global and local explainability on the dashboard.
- Build a REST API using FastAPI to serve model predictions to external CRM systems.
