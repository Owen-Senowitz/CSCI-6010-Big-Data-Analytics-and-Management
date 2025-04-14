import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
import time

# Start timing
start_time = time.time()

# Create a directory for saving models
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
# Load dataset
data = pd.read_csv("data/trip_data.csv")
print("Dataset loaded successfully!")

# Convert pickup datetime column to datetime objects
print("Converting datetime columns...")
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
print("Datetime column converted!")

# Extract features from datetime column
print("Extracting datetime features...")
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_dayofweek'] = data['pickup_datetime'].dt.dayofweek
data['pickup_month'] = data['pickup_datetime'].dt.month
print("Datetime features extracted!")

# Remove outliers and invalid trip durations
print("Cleaning data...")
data = data[(data['trip_duration'] > 0) & (data['trip_duration'] < 36000)]  # 0-10 hours
print(f"Data cleaned. Remaining rows: {data.shape[0]}")

# Define X and y
print("Preparing feature matrix and target variable...")
X = data[['pickup_hour', 'pickup_dayofweek', 'pickup_month',
          'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude']]
y = data['trip_duration']
print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Normalize input features
print("Normalizing input features...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
print("Input features normalized!")

# Train models
models = {
    # "decision_tree": DecisionTreeRegressor(),
    # "knn": KNeighborsRegressor(n_neighbors=5),
    "linear_regression": LinearRegression(),
    # "neural_network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, alpha=0.01),
    # "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "xgboost": XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
}

results = {}
print("Training and evaluating models...")
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "neural_network":
        model.fit(X_train_scaled, y_train)  # Neural network requires scaled target
    elif model_name == "linear_regression":
        model.fit(X_train, y_train)  # Linear regression does not need scaled input
    else:
        model.fit(X_train_scaled, y_train)
    print(f"{model_name} trained successfully!")

    # Predict on test set
    if model_name == "linear_regression":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {
        "mean_squared_error": mse,
        "r2_score": r2
    }
    print(f"{model_name} - Mean Squared Error: {mse:.2f}, R² Score: {r2:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red')
    plt.xlabel("Actual Trip Duration")
    plt.ylabel("Predicted Trip Duration")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.savefig(f"models/{model_name}_regression_plot.png")
    plt.close()

# Save models
print("Saving models to 'models' directory...")
for model_name, model in models.items():
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"{model_name} saved to models/{model_name}.pkl")

# Save scalers
print("Saving scalers...")
joblib.dump(scaler_X, "models/scaler_X.pkl")
print("Scaler saved!")

# Save test data
print("Saving test data for analysis...")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")
print("Test data saved!")

# Save results to JSON
print("Saving results to results.json...")
with open("models/results.json", "w") as results_file:
    json.dump(results, results_file, indent=4)
print("Results saved to results.json!")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training, evaluation, and saving completed successfully! Total time: {elapsed_time:.2f} seconds.")
