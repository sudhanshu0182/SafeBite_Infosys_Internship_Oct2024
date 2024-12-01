import os
from flask import Flask, request, jsonify
import joblib, pickle
import pandas as pd

# Initialize Flask app:
app = Flask(__name__)

# Loading the models:
with open("../Model/leave_one_out_encoder.pkl", "rb") as file:
    loaded_encoder = pickle.load(file)

with open("../Model/random_forest_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


@app.route('/')
def home():
    return "Welcome to the SafeBite - AI-powered Allergen Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])
        
        # Encode categorical columns
        categorical_columns = input_data.select_dtypes(include=['object']).columns
        input_data_encoded = loaded_encoder.transform(input_data[categorical_columns])
        input_data = pd.concat([input_data.drop(categorical_columns, axis=1), input_data_encoded], axis=1)

        # Make prediction using the trained model
        prediction = loaded_model.predict(input_data)
        result = "This product contains allergens" if prediction == 0 else "This product does not contain allergens"

        return jsonify(result=result)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    # app.run(debug=True)
        # Get the port from the environment variable (default is 5000)
    port = int(os.environ.get('PORT', 5000))
    # Run the app on 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=port, debug=True)
