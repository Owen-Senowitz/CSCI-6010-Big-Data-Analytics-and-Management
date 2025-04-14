import os
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
import time

start_time = time.time()

os.makedirs("models", exist_ok=True)

print("Connecting to database and loading data...")
conn = psycopg2.connect(
    dbname="taxi_data",
    user="owen",
    password="password123",
    host="localhost",
    port="5432"
)
query = """
SELECT pickup_datetime, pickup_longitude, pickup_latitude,
       dropoff_longitude, dropoff_latitude, trip_duration
FROM raw_trips
WHERE trip_duration > 0 AND trip_duration < 36000;
"""
data = pd.read_sql(query, conn)
conn.close()
print("Data loaded successfully from database!")

print("Converting datetime column...")
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

print("Extracting datetime features...")
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_dayofweek'] = data['pickup_datetime'].dt.dayofweek
data['pickup_month'] = data['pickup_datetime'].dt.month

print("Preparing feature matrix and target variable...")
X = data[['pickup_hour', 'pickup_dayofweek', 'pickup_month',
          'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude']]
y = data['trip_duration']

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Normalizing input features...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

models = {
    "decision_tree": DecisionTreeRegressor(),
    "knn": KNeighborsRegressor(n_neighbors=5),
    "linear_regression": LinearRegression(),
    "neural_network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, alpha=0.01),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "xgboost": XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
}

results = {}
print("Training and evaluating models...")
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "neural_network":
        model.fit(X_train_scaled, y_train)
    elif model_name == "linear_regression":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_scaled, y_train)
    print(f"{model_name} trained.")

    if model_name == "linear_regression":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {
        "mean_squared_error": mse,
        "r2_score": r2
    }
    print(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red')
    plt.xlabel("Actual Trip Duration")
    plt.ylabel("Predicted Trip Duration")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.savefig(f"models/{model_name}_regression_plot.png")
    plt.close()

print("Saving models...")
for model_name, model in models.items():
    joblib.dump(model, f"models/{model_name}.pkl")

print("Saving scaler and test data...")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Saving results to JSON...")
with open("models/results.json", "w") as results_file:
    json.dump(results, results_file, indent=4)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Done! Total time: {elapsed_time:.2f} seconds.")
