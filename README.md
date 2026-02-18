# Cyberzero Research Project: Phishing Attack Detection
Using Machine Learning to Secure the Web.

📌 Project Overview
The Cyberzero Research Project focuses on the development of a robust machine learning model designed to classify URLs and emails as either Phishing or Legitimate. By extracting key features and utilizing ensemble learning, this tool provides a first line of defense against social engineering attacks.

Objective: Build a high-precision classifier for malicious web entities.

Methodology: Automated feature extraction followed by Random Forest classification.

Outcome: A functional Command-Line Interface (CLI) tool for real-time URL analysis.

🎯 Objective
The primary goal is to identify the most effective features (lexical, textual, and metadata) and algorithms to achieve high-precision detection. We focus specifically on minimizing False Negatives—ensuring that phishing attempts do not go undetected.

🛠 Methodology
1. Data Collection & Sourcing
The model is trained using a combination of publicly available datasets:

Phishing Sources: PhishTank, OpenPhish, and Kaggle repositories.

Legitimate Sources: Reputable web directories and the Enron Email Dataset for textual analysis.

2. Feature Extraction
Raw data is converted into numerical vectors across three main categories:

URL-Based (Lexical) Features:

URL length and number of subdomains.

Presence of sensitive symbols (e.g., @ used to mask hostnames).

IP address usage instead of domain names.

HTTPS protocol verification.

Textual Features: Identifying urgency-based keywords like "Click here," "Verify," or "Urgent" using NLP.

Metadata & HTML: Analyzing header inconsistencies and hidden links.

3. Model Training
We utilize the Random Forest algorithm, an ensemble method that constructs multiple decision trees. This approach is highly effective at handling the high dimensionality of URL strings and providing stable predictions.

💻 Implementation & Usage
Prerequisites
Python 3.x

A dataset named phishing.csv (sourced from Kaggle) in the project root.

Setup Instructions
Create a virtual environment:

Bash
python -m venv environment
Activate the environment:

Windows: environment\Scripts\activate.bat

Linux/Mac: source environment/bin/activate

Install dependencies:

Bash
pip install pandas scikit-learn
Running the Tool
Execute the script by passing a URL as an argument:

Bash
python phishing-detect.py https://example-phishing-site.com
📄 Source Code (phishing-detect.py)
Python
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
import sys
import os

def extract_all_features(url, feature_names):
    """Simplified feature extraction for demonstration."""
    features = {
        'url_length': len(url),
        'has_at_symbol': '@' in url,
        'num_subdomains': url.count('.'),
        'has_ip_address': bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)),
        'has_https': url.startswith('https')
    }
    features_df = pd.DataFrame([features])
    return features_df[feature_names]

def train_model():
    dataset_file = 'phishing.csv' 
    if not os.path.exists(dataset_file):
        print(f"Error: '{dataset_file}' not found. Please add a Kaggle dataset to the directory.")
        sys.exit(1)

    df = pd.read_csv(dataset_file)
    # Selecting simplified features for the demo
    df = df[['url_length', 'has_at_symbol', 'num_subdomains', 'has_ip_address', 'has_https', 'Label']]
    
    X = df.drop(columns=['Label'])
    y = df['Label'].replace(-1, 0) # Normalize labels
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phishing-detect.py <URL>")
        sys.exit(1)

    # Initialize and Predict
    model, feature_names = train_model()
    url_to_check = sys.argv[1]
    
    features_df = extract_all_features(url_to_check, feature_names)
    prediction = model.predict(features_df)
    result = "Phishing" if prediction[0] == 1 else "Legitimate"

    print("\n--- Phishing Detector ---")
    print(f"URL: {url_to_check}")
    print(f"Prediction: {result}")
    
    if result == "Phishing":
        print("⚠️ Warning: This URL is likely a PHISHING attempt!")
    else:
        print("✅ This URL appears to be legitimate.")
