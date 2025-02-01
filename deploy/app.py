import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Habitable Planet Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get input data as JSON
    features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
    prediction = model.predict(features)  # Make prediction
    return jsonify({"P_HABITABLE": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app
