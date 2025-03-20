from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
model = pickle.load(open("finalized_maternal_model.sav", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get features from request
        features = [float(request.json['features'][i]) for i in range(5)]
        features_array = np.array([features]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]  
        print(prediction)

        # Map to risk levels
        risk_mapping = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
        risk_level = risk_mapping.get(prediction, "Unknown Risk")

        return jsonify({"prediction": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
