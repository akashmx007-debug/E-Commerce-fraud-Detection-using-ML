import os
import pickle
import subprocess

MODEL_PATH = "fraud_stack.pkl"
MODEL_ID = "1ij6SsISOl4zL0zJL0mA2qytLibEvNW5R"  # ✅ use only the file ID

def download_model():
    """Download model file from Google Drive using gdown"""
    print("⚙️ Model file not found. Downloading from Google Drive using gdown...")
    try:
        subprocess.run(["pip", "install", "--quiet", "gdown"], check=True)
        subprocess.run(["gdown", "--id", MODEL_ID, "-O", MODEL_PATH], check=True)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise

# --- Load Stacking Model ---
if not os.path.exists(MODEL_PATH):
    download_model()

try:
    with open(MODEL_PATH, "rb") as f:
        stacking = pickle.load(f)
    print("✅ Stacking model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load stacking model: {e}")
    if os.path.getsize(MODEL_PATH) < 1000:
        print("⚠️ The downloaded file seems invalid (too small). Try manual download.")
    raise

# --- Load XGBoost Model ---
try:
    xgboost = pickle.load(open("fraud_xg.pkl", "rb"))
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load XGBoost model: {e}")
    raise

import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
import pandas as pd

app = Flask(__name__)  # Initialize the flask App

xgboost = pickle.load(open('fraud_xg.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Transaction_Amount = request.form['Transaction_Amount']
        Payment_Method = request.form['Payment_Method']
        if Payment_Method == '0':
            Pay = 'PayPal'
        elif Payment_Method == '1':
            Pay = 'credit card'
        elif Payment_Method == '2':
            Pay = 'debit card'
        elif Payment_Method == '3':
            Pay = 'bank transfer'
        Product_Category = request.form['Product_Category']
        if Product_Category == '0':
            prod = 'electronics'
        elif Product_Category == '1':
            prod = 'toys & games'
        elif Product_Category == '2':
            prod = 'clothing'
        elif Product_Category == '3':
            prod = 'home & garden'
        elif Product_Category == '4':
            prod = 'health & beauty'
        Quantity = request.form['Quantity']
        Customer_Age = request.form['Customer_Age']
        Device_Used = request.form['Device_Used']
        if Device_Used == '0':
            Devi = 'desktop'
        elif Device_Used == '1':
            Devi = 'tablet'
        elif Device_Used == '2':
            Devi = 'mobile'
        Account_Age_Days = request.form['Account_Age_Days']
        Transaction_Hour = request.form['Transaction_Hour']
        Address_Match = request.form['Address_Match']
        if Address_Match == '0':
            Address = 'No'
        elif Address_Match == '1':
            Address = 'Yes'
        Model = request.form['Model']

        input_variables = pd.DataFrame([[Transaction_Amount, Payment_Method, Product_Category, Quantity, Customer_Age, Device_Used, Account_Age_Days, Transaction_Hour, Address_Match]],
                                       columns=['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity', 'Customer Age', 'Device Used', 'Account Age Days', 'Transaction Hour', 'Address Match'],
                                       index=['input'])

       
        input_variables['Transaction Amount'] = input_variables['Transaction Amount'].astype(float)
        input_variables['Quantity'] = input_variables['Quantity'].astype(float)
        input_variables['Customer Age'] = input_variables['Customer Age'].astype(float)
        input_variables['Account Age Days'] = input_variables['Account Age Days'].astype(float)
        input_variables['Transaction Hour'] = input_variables['Transaction Hour'].astype(float)
        
    
        input_variables['Payment Method'] = input_variables['Payment Method'].astype(int)
        input_variables['Product Category'] = input_variables['Product Category'].astype(int)
        input_variables['Device Used'] = input_variables['Device Used'].astype(int)
        input_variables['Address Match'] = input_variables['Address Match'].astype(int)

        print(input_variables)

        if Model == 'XGBClassifier':
            prediction = xgboost.predict(input_variables)
            outputs = prediction[0]
        elif Model == 'StackingClassifier':
            prediction = stacking.predict(input_variables)
            outputs = prediction[0]

        if outputs == 1:
            results = "Fraudulent"
        else:
            results = "Not Fraudulent"

    return render_template('result.html', prediction_text=results, model=Model, Transaction_Amount=Transaction_Amount, Pay=Pay, prod=prod, Quantity=Quantity, Customer_Age=Customer_Age, Devi=Devi, Account_Age_Days=Account_Age_Days, Transaction_Hour=Transaction_Hour, Address=Address)

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
