from UserInputData import UserInput
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

model_path = 'model_linear.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.get('/')
def read_root():
    return {'message': 'Dynamic Ride Pricing Model API'}

@app.post('/predict')
def predict_banknote(data: UserInput):
    input_data = {
        'distance': data.distance,
        'name_Black': data.name_Black,
        'name_Black SUV': data.name_BlackSUV,
        'name_Lux': data.name_Lux,
        'name_Lux Black': data.name_LuxBlack,
        'name_Lux Black XL': data.name_LuxBlackXL,
        'name_Lyft': data.name_Lyft,
        'name_Lyft XL': data.name_LyftXL,
        'name_Shared': data.name_Shared,
        'name_UberPool': data.name_UberPool,
        'name_UberX': data.name_UberX,
        'name_UberXL': data.name_UberXL,
        'name_WAV': data.name_WAV,
        'cab_type_Lyft': data.cab_type_Lyft,
        'cab_type_Uber': data.cab_type_Uber
    }

    prediction = model.predict([list(input_data.values())])

    return {"prediction": prediction}

if __name__ == "__main__":
    app.run(debug=True)