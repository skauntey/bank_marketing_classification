import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import json
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__, template_folder ='templates')



cols_input = pickle.load(open('columns.pkl', 'rb'))


def encode_data(df):
    col_toEncode = ['job', 'marital', 'month', 'poutcome']
    col_toOneHot = ['education', 'default', 'housing', 'loan', 'contact']
    
    def label_encoder(df):
        le = LabelEncoder()
        labelenconded = le.fit(df)
        df = labelenconded.transform(df)
        return df
    
    for i in col_toEncode:
        df[i] = label_encoder(df[i])
    df = pd.get_dummies(data = df, columns=col_toOneHot, drop_first = False)
    df = df.sort_index(axis=1)
    return df

def standardize_data(dta):
    scaler = joblib.load("std_scaler.pkl")
    X_transformed = scaler.fit_transform(dta)
    return X_transformed



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    form_data = request.form.to_dict()
    print(form_data)
    
    df_input = pd.DataFrame.from_records([form_data],)
    df_input = df_input.drop(['submitBtn'], axis=1)
    df_input = pd.DataFrame(df_input)
    
    sample_df = pd.DataFrame(columns = cols_input)
    clean_df = encode_data(df_input)
    main_df = sample_df.append(clean_df)
    main_df = main_df.fillna(0)

    std_df = standardize_data(main_df)
    
    clf = joblib.load(open('best_classifier.joblib', 'rb'))
    pred = clf.predict(std_df)
    print(pred)
    
    '''
    For rendering results on HTML GUI
    '''
    
    output = pred[0]
    if output == 0:
        pt = 'No, customer will Not subscribe to the product.'
    elif output == 1:
        pt = 'Yes, customer will subscribe to the product.'
    else:
        pt = 'error'
        
    return render_template('predict.html', predicted_value=f"Customer Churn rate: {pt}")
    #return render_template('index.html', prediction_text= pt)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)