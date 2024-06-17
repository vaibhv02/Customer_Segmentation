from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl', 'rb'))

def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M')
    max_date = max(retail['InvoiceDate'])
    retail['Difference'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Difference'].min().reset_index()
    rfm_p['Difference'] = rfm_p['Difference'].dt.days
    rfm = pd.merge(rfm_m, rfm_p, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_f, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Recency', 'Frequency']
    

    Q1 = rfm[['Amount', 'Recency', 'Frequency']].quantile(0.25)
    Q3 = rfm[['Amount', 'Recency', 'Frequency']].quantile(0.75)
    IQR = Q3 - Q1
    rfm = rfm[~((rfm < (Q1 - 1.5 * IQR)) | (rfm > (Q3 + 1.5 * IQR))).any(axis=1)]

    return rfm

def preprocessing_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns=['Amount', 'Frequency', 'Recency'])
    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df_with_id, df = preprocessing_data(file_path)
    results = model.predict(df)
    df_with_id['Cluster_ID'] = results

    sns.stripplot(x='Cluster_ID', y='Amount', data=df_with_id, hue='Cluster_ID')
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_ID', y='Frequency', data=df_with_id, hue='Cluster_ID')
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_ID', y='Recency', data=df_with_id, hue='Cluster_ID')
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    response = {'amount_img': amount_img_path,
                'freq_img': freq_img_path,
                'recency_img': recency_img_path}
    return jsonify(response)

    

if __name__ == "__main__":
    app.run(debug=True)
