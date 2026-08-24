import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Churn Prediction: Exploratory Data Analysis\n",
    "\n",
    "This notebook covers data loading, cleaning, EDA, feature engineering, and a basic modeling approach."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading & Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../data/raw/Telco-Customer-Churn.csv')\n",
    "df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)\n",
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])\n",
    "df['TotalCharges'] = df['TotalCharges'].fillna(0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Churn Rate\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x='Churn', data=df)\n",
    "plt.title('Overall Churn Rate')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(os.path.join('notebooks', '01_churn_analysis.ipynb'), 'w') as f:
    json.dump(notebook, f, indent=1)
