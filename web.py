from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import os
from io import BytesIO

app = Flask(__name__)

# Load the encoders, scaler, and model
with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

with open('ordinalencoder.pkl', 'rb') as file:
    ordinalencoder = pickle.load(file)

with open('freq_map.pkl', 'rb') as file:
    loaded_freq_maps = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    scaler1 = pickle.load(file)

with open('xgb_model4.pkl', 'rb') as file:
    xgb_model4 = pickle.load(file)

with open('encoder1.pkl', 'rb') as file:
    label_encoder = pickle.load(file)



# Columns and features used for training and scaling
traincolumns = ['BeneID', 'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode',
                'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
                'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
                'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
                'RenalDiseaseIndicator', 'Provider_PRV51459', 'Provider_PRV51574',
                'Provider_PRV53797', 'Provider_PRV53918', 'Provider_PRV54895',
                'Provider_PRV55215', 'Provider_lessfrequent', 'ClmProcedureCode_1',
                'ClmProcedureCode_2', 'Race', 'State', 'County', 'NoOfMonths_PartACov',
                'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
                'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                'ChronicCond_ObstrPulmonary', 'ChronicCond_Diabetes',
                'ChronicCond_IschemicHeart', 'ChronicCond_rheumatoidarthritis',
                'ChronicCond_stroke', 'Claimduration', 'TotalClaimAmt', 'isinpatient',
                'PhysicianOverlap', 'PhysicianInfoMissing',
                'AttendingPhysicianFrequency', 'OperatingPhysicianFrequency',
                'OtherPhysicianFrequency', 'PhysicianRoleCount', 'TotalLengthOfStay',
                'Age', 'TotalAnnualIPAmount']

numerical_features = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'Race', 'State', 'County',
                      'NoOfMonths_PartACov', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
                      'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
                      'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_rheumatoidarthritis',
                      'ChronicCond_stroke', 'Claimduration', 'TotalClaimAmt', 'isinpatient', 'PhysicianOverlap',
                      'PhysicianInfoMissing', 'AttendingPhysicianFrequency', 'OperatingPhysicianFrequency',
                      'OtherPhysicianFrequency', 'PhysicianRoleCount', 'TotalLengthOfStay', 'Age', 'TotalAnnualIPAmount']

# Initialize a global variable for test data
test_data = None

# Home route
@app.route('/')
def home():
    print("Home route called")
    return render_template('home.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    global test_data
    try:
        print("Request.files:", request.files)  # Debugging: Check the content of request.files
        if 'file' not in request.files:
            return "No file part in the request", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if not file.filename.endswith('.csv'):
            return "Only CSV files are allowed", 400

        # Read the uploaded CSV file
        test_data = pd.read_csv(file)
        df = test_data.copy()
       # Load the mapping files
        provider_mapping = pd.read_csv('provider_mapping.csv')
        target_encoding_mapping = pd.read_csv('beneid_targetencoded.csv')
        global_mean = 0.3812106891480103
 

        # Preprocessing functions
        def encode_and_align(data, column, train_columns, encoder):
            encoded = pd.DataFrame(encoder.transform(data[[column]]), columns=train_columns)
            for col in train_columns:
                if col not in encoded.columns:
                    encoded[col] = 0
            encoded = encoded[train_columns]
            return encoded

        def apply_target_encoding(data, column_name, target_mapping, global_mean):
            target_mapping = target_mapping.drop_duplicates(subset=column_name, keep='first')
            data['TargetEncoded'] = data['BeneID'].map(target_mapping.set_index('BeneID')['TargetEncoded'])
            data['TargetEncoded'] = data['TargetEncoded'].fillna(global_mean)
            return data

        # Perform encoding, scaling, and transformations
        # Drop duplicate rows based on the 'Provider' column, keeping the first occurrence
        provider_mapping = provider_mapping.drop_duplicates(subset=['Provider'], keep='first')
        df['provider_labeled'] = df['Provider'].map(provider_mapping.set_index('Provider')['provider_labeled'])
        df = df.drop('Provider', axis=1)
        df = df.rename(columns={'provider_labeled': 'Provider'})
        

        train_columns = ['Provider_PRV51459', 'Provider_PRV51574', 'Provider_PRV53797', 'Provider_PRV53918', 'Provider_PRV54895', 'Provider_PRV55215', 'Provider_lessfrequent']
        df1 = encode_and_align(df, 'Provider', train_columns, onehotencoder)
        df = pd.concat([df, df1], axis=1)
        df = df.drop(['Provider'], axis=1)

        df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].replace({'Y': 1}).astype(int)
        df['DiagnosisGroupCode'] = df['DiagnosisGroupCode'].astype(str)
        df['DiagnosisGroupCode'] = ordinalencoder.transform(df[['DiagnosisGroupCode']])

        columns_to_encode = ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 
                             'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 
                             'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9']
        for col in columns_to_encode:
            df[col] = df[col].map(loaded_freq_maps).fillna(0)

        df = apply_target_encoding(df, 'BeneID', target_encoding_mapping, global_mean)
        df['BeneID'] = df['TargetEncoded']
        df = df.drop(['TargetEncoded'], axis=1)
        df[numerical_features] = scaler1.transform(df[numerical_features])
        df = df[traincolumns]

        # Model prediction
        predictions = xgb_model4.predict(df)
        predictions_labels = label_encoder.inverse_transform(predictions)
        test_data['PredictedValue'] = predictions_labels

        # Display the result
        table_html = test_data.to_html(classes='table table-striped', index=False)
        #print("Generated HTML table:", table_html)  # For debugging
        #return render_template('result.html', table=table_html, download_path='/download')
        return render_template('result.html', table_html=table_html, download_path='/download')

        

    except Exception as e:
        return f"Error: {e}", 500

# Route for downloading results
@app.route('/download', methods=['GET'])
def download_csv():
    global test_data
    try:
        csv_data = test_data.to_csv(index=False)
        # Convert the CSV string into binary format
        binary_data = BytesIO(csv_data.encode())  # Encoding string to binary
        binary_data.seek(0)  # Rewind the binary data for reading
        
        return send_file(binary_data, mimetype='text/csv', as_attachment=True, download_name='result.csv')
        #return send_file(io.StringIO(csv_data), mimetype='text/csv', as_attachment=True, download_name='result.csv')
    except Exception as e:
        return f"Error generating CSV: {e}", 500

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, port=8080)

