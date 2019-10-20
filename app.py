from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/pingu"
mongo = PyMongo(app)
model = pickle.load(open('model.pickle', 'rb'))

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    arr = np.array([data['operator'], data['network'], data['rating'], data['latitude'], data['longitude']]).reshape(1, -1)
    print(arr)
    prediction = model.predict(arr)
    return jsonify({"prediction": int(prediction[0])})


@app.route('/capture', methods=['POST'])
def capture_location():
    data = request.get_json(force=True)
    mongo.db.locations.insert_one(data)
    return jsonify({'Status': 'Location Captured'})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
