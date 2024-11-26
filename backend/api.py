from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load model
model = joblib.load(os.path.join(current_dir, 'iris_model.pkl'))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    ]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'class': prediction[0]})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5001', debug=True)
