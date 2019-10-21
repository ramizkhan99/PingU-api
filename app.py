from flask import Flask, request, jsonify, send_file, redirect
import numpy as np
import pickle, csv, os, json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pickle', 'rb'))

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    arr = np.array([data['operator'], data['network'], data['rating'], data['latitude'], data['longitude']]).reshape(1, -1)
    print(arr)
    prediction = model.predict(arr)
    return jsonify({"prediction": int(prediction[0])})


# @app.route('/capture', methods=['POST'])
# def capture_location():
#     data = request.get_json(force=True)
#     mongo.db.locations.insert_one(data)
#     return jsonify({'Status': 'Location Captured'})


@app.route('/download', methods=['GET'])
def download_csv():
    download = os.path.join(os.getcwd(), 'PingU-dataset.csv')
    return send_file(download, as_attachment=True)


@app.route('/generate', methods=['POST'])
def json_to_csv():
    locations = request.get_json(force=True)
    csv_file = csv.writer(open('test.csv', 'w', newline=''))
    csv_file.writerow(["carrier",
                       "speed",
                       "lat",
                       "lon"])
    for key in locations.keys():
        location = locations[key]
        csv_file.writerow([location['carrier'],
                           location['speed'],
                           location['lat'],
                           location['lon']])
    return redirect('/download')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
