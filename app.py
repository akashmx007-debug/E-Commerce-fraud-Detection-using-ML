import os
import pickle
import subprocess
import pandas as pd
from flask import Flask, request, render_template

# -----------------------------
# Model Setup
# -----------------------------
MODEL_STACK_URL = "https://drive.google.com/uc?id=1ij6SsISOl4zL0zJL0mA2qytLibEvNW5R"
MODEL_XGB_URL = "https://drive.google.com/uc?id=<YOUR_XGBOOST_ID>"

MODEL_STACK_FILE = "fraud_stack.pkl"
MODEL_XGB_FILE = "fraud_xg.pkl"

def download_model(url, filename):
    """Download model from Google Drive if not exists."""
    if not os.path.exists(filename):
        try:
            print(f"⚙️ Downloading {filename} from Google Drive...")
            subprocess.run(["pip", "install", "gdown"], check=True)
            subprocess.run(["gdown", url, "-O", filename], check=True)
            print(f"✅ {filename} downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

# Download models
download_model(MODEL_STACK_URL, MODEL_STACK_FILE)
download_model(MODEL_XGB_URL, MODEL_XGB_FILE)

# Load models
try:
    with open(MODEL_STACK_FILE, "rb") as f:
        stacking = pickle.load(f)
    print("✅ Stacking model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load stacking model: {e}")

try:
    with open(MODEL_XGB_FILE, "rb") as f:
        xgboost = pickle.load(f)
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load XGBoost model: {e}")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Convert inputs to a dataframe
    input_df = pd.DataFrame([[
        float(data['Transaction_Amount']),
        int(data['Payment_Method']),
        int(data['Product_Category']),
        float(data['Quantity']),
        float(data['Customer_Age']),
        int(data['Device_Used']),
        float(data['Account_Age_Days']),
        float(data['Transaction_Hour']),
        int(data['Address_Match'])
    ]], columns=[
        'Transaction Amount', 'Payment Method', 'Product Category', 'Quantity',
        'Customer Age', 'Device Used', 'Account Age Days', 'Transaction Hour', 'Address Match'
    ])

    model_choice = data['Model']
    if model_choice == 'XGBClassifier':
        pred = xgboost.predict(input_df)[0]
    else:
        pred = stacking.predict(input_df)[0]

    result = "Fraudulent" if pred == 1 else "Not Fraudulent"
    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    # Run on Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

