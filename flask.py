from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your saved models
LR = joblib.load('models/LogisticRegression.pkl')
DT = joblib.load('models/DecisionTree.pkl')
GBC = joblib.load('models/GradientBoostingClassifier.pkl')
RFC = joblib.load('models/RandomForestClassifier.pkl')
RNN_model = tf.keras.models.load_model('models/RNN_model.h5')
LSTM_model = tf.keras.models.load_model('models/LSTM_model.h5')
BiLSTM_model = tf.keras.models.load_model('models/BiLSTM_model.h5')

# Load the vectorization model
vectorization = joblib.load('Models/vectorization.pkl')  # Ensure you save the vectorization model

app = Flask(__name__)

def output_label(label):
    return "Real" if label == 1 else "Fake"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(lambda x: x)  # Placeholder for any preprocessing function
    new_x_test = new_def_test["text"]

    # Vectorization for classical models
    new_xv_test = vectorization.transform(new_x_test)

    # Make predictions with all models
    results = {}

    # Classical models
    results['Logistic Regression'] = output_label(LR.predict(new_xv_test)[0])
    results['Decision Tree'] = output_label(DT.predict(new_xv_test)[0])
    results['Gradient Boosting'] = output_label(GBC.predict(new_xv_test)[0])
    results['Random Forest'] = output_label(RFC.predict(new_xv_test)[0])

    # RNN model
    new_xv_test_rnn = new_xv_test.toarray().reshape((new_xv_test.shape[0], 1, new_xv_test.shape[1]))
    results['RNN'] = output_label((RNN_model.predict(new_xv_test_rnn) > 0.5).astype(int)[0][0])

    # LSTM model
    new_xv_test_lstm = new_xv_test.toarray().reshape((new_xv_test.shape[0], 1, new_xv_test.shape[1]))
    results['LSTM'] = output_label((LSTM_model.predict(new_xv_test_lstm) > 0.5).astype(int)[0][0])

    # BiLSTM model
    new_xv_test_bilstm = new_xv_test.toarray().reshape((new_xv_test.shape[0], 1, new_xv_test.shape[1]))
    results['BiLSTM'] = output_label((BiLSTM_model.predict(new_xv_test_bilstm) > 0.5).astype(int)[0][0])

    return render_template('index.html', results=results, news=news)

if __name__ == '__main__':
    app.run(debug=True)
