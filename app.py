from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os  # Import os for PORT handling

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
model = pickle.load(open("finalized_maternal_model.sav", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get features from request
        data = request.json
        features = [float(data['features'][i]) for i in range(5)]
        features_array = np.array([features]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]  

        # Map to risk levels
        risk_mapping = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
        risk_level = risk_mapping.get(prediction, "Unknown Risk")

        return jsonify({"prediction": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Use Render's PORT
    app.run(host="0.0.0.0", port=port)
