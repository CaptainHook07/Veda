import urllib.request
import os

def download_telco_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Telco-Customer-Churn.csv")
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_telco_data()
