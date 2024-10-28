from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from pymongo import MongoClient  # assuming MongoDB for storage (optional)

app = Flask(__name__)
CORS(app)

# Load the SVM model
try:
    with open("models\svm_model.pkl", 'rb') as f:
        svm_model = pickle.load(f)
    print("SVM model loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")

# Example database setup (optional)
client = MongoClient("mongodb://localhost:27017/")
db = client["phone_db"]
phone_collection = db["phone_specs"]

# Route for phone comparison prediction
@app.route('/compare', methods=['POST'])
def compare_phones():
    data = request.json
    try:
        # Ensure that the expected features are provided in the request
        features = [data.get('feature1'), data.get('feature2'), data.get('feature3')]
        prediction = svm_model.predict([features])
        return jsonify({"comparison_result": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Route for similar phone suggestions
@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    # Retrieve all phone specs from the database (optional)
    try:
        phone_specs = list(phone_collection.find())
        return jsonify(phone_specs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for value prediction based on specs
@app.route('/predict_value', methods=['POST'])
def predict_value():
    data = request.json
    try:
        # Ensure the expected fields are provided
        specs = [data.get('price'), data.get('battery'), data.get('longevity')]
        value_prediction = svm_model.predict([specs])
        return jsonify({"value_prediction": value_prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

