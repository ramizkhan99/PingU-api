from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))

# Routes
@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    arr = np.array([data['operator'], data['network'], data['rating'], data['latitude'], data['longitude']]).reshape(1, -1)
    print(arr)
    prediction = model.predict(arr)
    return jsonify({"prediction": str(prediction[0])})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
