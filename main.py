from UserInputData import UserInput
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib

model_path = 'model_linear.pkl'
with open('model_linear.pkl', 'rb') as file:
    model = joblib.load(file)  # Use joblib.load instead of pickle.load

print("Type of the model: ", type(model))

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://dynamic-ride-pricing.vercel.app"])

@app.route('/', methods=['GET'])
def read_root():
    return {'message': 'Dynamic Ride Pricing Model API'}

def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    data = request.get_json()

    input_data = {
        'distance': data['distance'],
        'name_Black': data['name_Black'],
        'name_Black SUV': data['name_BlackSUV'],
        'name_Lux': data['name_Lux'],
        'name_Lux Black': data['name_LuxBlack'],
        'name_Lux Black XL': data['name_LuxBlackXL'],
        'name_Lyft': data['name_Lyft'],
        'name_Lyft XL': data['name_LyftXL'],
        'name_Shared': data['name_Shared'],
        'name_UberPool': data['name_UberPool'],
        'name_UberX': data['name_UberX'],
        'name_UberXL': data['name_UberXL'],
        'name_WAV': data['name_WAV'],
        'cab_type_Lyft': data['cab_type_Lyft'],
        'cab_type_Uber': data['cab_type_Uber']
    }

    # Ensure that the prediction is a list or single value
    prediction = model.predict([list(input_data.values())])

    print("Prediction: ", prediction[0][0])
    # Return JSON serializable response
    return jsonify({"prediction": prediction[0][0]})  # Convert prediction to a scalar or JSON serializable type

if __name__ == "__main__":
    app.run(debug=True)
