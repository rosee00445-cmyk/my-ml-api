from flask import Flask, request, jsonify
from flask_cors import CORS  # This is critical for Wix to talk to your API
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows your Wix domain to access this API

# Load your model once when the server starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the data sent from Wix
        data = request.get_json() 
        
        # 2. Extract features (adjust the keys to match your sensor data)
        # Assuming Wix sends: {"features": [val1, val2, val3...]}
        features = np.array(data['features']).reshape(1, -1)
        
        # 3. Make the prediction
        prediction = model.predict(features)
        
        # 4. Send the result back to Wix
        return jsonify({
            'status': 'success',
            'prediction': str(prediction[0])
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
