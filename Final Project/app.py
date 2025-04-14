from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)

# Load models and scaler
models = {
    # "decision_tree": joblib.load("models/decision_tree.pkl"),
    # "knn": joblib.load("models/knn.pkl"),
    "linear_regression": joblib.load("models/linear_regression.pkl"),
    # "neural_network": joblib.load("models/neural_network.pkl"),
    # "random_forest": joblib.load("models/random_forest.pkl"),
    # "xgboost": joblib.load("models/xgboost.pkl")
}
scaler = joblib.load("models/scaler_X.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from JSON
        data = request.json
        pickup_datetime = data["pickup_datetime"]
        pickup_longitude = float(data["pickup_longitude"])
        pickup_latitude = float(data["pickup_latitude"])
        dropoff_longitude = float(data["dropoff_longitude"])
        dropoff_latitude = float(data["dropoff_latitude"])

        # Handle datetime parsing using fromisoformat (Python 3.7+)
        try:
            pickup_datetime_obj = datetime.fromisoformat(pickup_datetime.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Invalid datetime format"}), 400

        # Extract datetime features
        pickup_hour = pickup_datetime_obj.hour
        pickup_dayofweek = pickup_datetime_obj.weekday()
        pickup_month = pickup_datetime_obj.month

        # Prepare features
        features = pd.DataFrame([{
            "pickup_hour": pickup_hour,
            "pickup_dayofweek": pickup_dayofweek,
            "pickup_month": pickup_month,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude
        }])

        # Scale features
        features_scaled = scaler.transform(features)

        # Generate predictions
        predictions = {
            model_name: float(model.predict(features_scaled if model_name != "linear_regression" else features)[0])
            for model_name, model in models.items()
        }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/results", methods=["GET"])
def results():
    try:
        # Load results from results.json
        with open("models/results.json", "r") as results_file:
            results = json.load(results_file)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/heatmap-data")
def heatmap_data():
    import psycopg2

    try:
        conn = psycopg2.connect(
            dbname="taxi_data",
            user="owen",
            password="password123",
            host="localhost",
            port="5432"
        )
        query = """
            SELECT pickup_latitude, pickup_longitude
            FROM raw_trips
            WHERE pickup_latitude IS NOT NULL AND pickup_longitude IS NOT NULL
              AND pickup_latitude BETWEEN 40.5 AND 41
              AND pickup_longitude BETWEEN -74.3 AND -73.5
            ORDER BY pickup_datetime DESC
            LIMIT 100000;
        """
        df = pd.read_sql(query, conn)
        conn.close()

        heatmap_points = df[['pickup_latitude', 'pickup_longitude']].values.tolist()
        return jsonify(heatmap_points)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
