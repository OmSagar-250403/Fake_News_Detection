# Fake News Detection System

This project is a Flask-based web application for detecting fake news articles using multiple machine learning models. It utilizes a user-friendly interface that allows users to input news articles and receive predictions on whether the news is fake or real.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Saving Models](#saving-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- Input news articles and receive predictions on their authenticity.
- Implements various classification models for comparison.
- Displays clear output on the webpage indicating whether the news is "Real" or "Fake."

## Technologies Used

- **Backend:** Python, Flask
- **Machine Learning Models:** Logistic Regression, Decision Tree, Gradient Boosting Classifier, Random Forest, RNN, LSTM, BiLSTM
- **Data Processing:** Pandas, Scikit-learn
- **Deep Learning Framework:** TensorFlow
- **Frontend:** HTML, CSS (Bootstrap for styling)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection


2. pip install -r requirements.txt


3. Run the main.ipynb notebook to train your models and save them in the Models directory. This notebook will generate the following files:

vectorization.pkl
LogisticRegression.pkl
DecisionTree.pkl
GradientBoostingClassifier.pkl
RandomForestClassifier.pkl
RNN_model.h5
LSTM_model.h5
BiLSTM_model.h5


4. run python app.py

5. Open your web browser and navigate to http://127.0.0.1:5000 to access the application.

Enter a news article in the text area and click "Predict" to see the result.

