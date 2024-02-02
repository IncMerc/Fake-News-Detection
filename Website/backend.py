from flask import Flask, request, jsonify
import pickle
import numpy as np  # Assuming you're using a model that requires NumPy
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Load your machine learning model and other necessary data
import re
import string
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
# Define a route to handle API requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)



        # Extract features from the data
        features = extract_features(data)
        # Make predictions using the model

        print(features)

        with open('your_model_LR.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('your_model_vec.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        
        vec = vectorizer.transform(features)
        
        prediction = model.predict(vec)

        
        print(prediction)
        # Convert the prediction to a JSON response
        response = {'prediction': prediction.tolist()}
        
        # print(response[0])      

        
        print("Check ")
        

        return jsonify({'Prediction': response})

    except Exception as e:
        print("Exdeption")

        print(e)
        return jsonify({'error': str(e)}), 400

def extract_features(data):
    # Implement your feature extraction logic here based on the data structure
    # For example, assuming the data is a list of values:
    testing_news = {"text":[data['features']]}

    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    #new_xv_test = vectorization.transform(new_x_test)
    # features = vectorization.transform()
    return new_def_test

if __name__ == '__main__':
    app.run(debug=True,  port=6969)
