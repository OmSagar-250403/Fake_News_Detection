from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import os
import string

# Initialize Flask app
app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Load your saved models
vectorization = joblib.load('Models/vectorization.pkl')
LR = joblib.load('Models/LogisticRegression.pkl')
DT = joblib.load('Models/DecisionTree.pkl')
GBC = joblib.load('Models/GradientBoostingClassifier.pkl')
RFC = joblib.load('Models/RandomForestClassifier.pkl')


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

    # Predictions with classical models
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    # Return combined prediction (you can update this logic to suit your needs)
    final_pred = max([pred_LR[0], pred_DT[0], pred_GBC[0], pred_RFC[0]])

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
