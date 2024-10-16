from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model
import re
import os
import string
import math

# Initialize Flask app
app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load your saved models
vectorization = joblib.load('Models/vectorization.pkl')
LR = joblib.load('Models/LogisticRegression.pkl')
DT = joblib.load('Models/DecisionTree.pkl')
GBC = joblib.load('Models/GradientBoostingClassifier.pkl')
RFC = joblib.load('Models/RandomForestClassifier.pkl')
RNN_model = load_model('Models/RNN_model.h5')
LSTM_model = load_model('Models/LSTM_model.h5')
BiLSTM_model = load_model('Models/BiLSTM_model.h5')


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def output_label(label):
    return "Fake" if label == 0 else "Real"


def manual_testing(news):
    # Prepare the input data
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)

    # Preprocess the input
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]

    # Vectorization for classical models
    new_xv_test = vectorization.transform(new_x_test)



    # Prepare input for RNN, LSTM, BiLSTM models
    new_xv_test_rnn = new_xv_test.toarray().reshape((new_xv_test.shape[0], 1, new_xv_test.toarray().shape[1]))
    # Make predictions with deep learning models
    pred_RNN = RNN_model.predict(new_xv_test_rnn)
    pred_RNN_label = (pred_RNN > 0.5).astype(int)



    new_xv_test_lstm = new_xv_test.toarray().reshape((new_xv_test.shape[0], 1, new_xv_test.toarray().shape[1]))
    pred_LSTM = LSTM_model.predict(new_xv_test_lstm)
    pred_LSTM_label = (pred_LSTM > 0.5).astype(int)



    new_xv_test_bilstm = new_xv_test.toarray()
    array_size = new_xv_test_bilstm.size
    batch_size = new_xv_test_bilstm.shape[0]
    
    # Find optimal shape for BiLSTM
    def find_reshape_shape(array_size, batch_size):
        total_features = array_size // batch_size
        factors = [(i, total_features // i) for i in range(1, int(math.sqrt(total_features)) + 1) if total_features % i == 0]
        return factors

    reshape_factors = find_reshape_shape(array_size, batch_size)
    reshape_timesteps, reshape_features = reshape_factors[-1]
    new_xv_test_bilstm = new_xv_test_bilstm.reshape((new_xv_test_bilstm.shape[0], reshape_timesteps, reshape_features))

    pred_BiLSTM = BiLSTM_model.predict(new_xv_test_bilstm)
    pred_BiLSTM_label = (pred_BiLSTM > 0.5).astype(int)

    # Predictions with classical models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    # Return combined prediction (you can update this logic to suit your needs)
    final_pred = max([pred_LR[0], pred_DT[0], pred_GBC[0], pred_RFC[0], 
                      pred_RNN_label[0][0], pred_LSTM_label[0][0], pred_BiLSTM_label[0][0]])
    
    
    return "FAKE" if final_pred == 0 else "REAL"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        predictions = manual_testing(message)
        
        # Assuming 'predictions' returns the final label (e.g., 'Fake' or 'Real')
        return render_template('index.html', prediction=predictions)
    else:
        return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
