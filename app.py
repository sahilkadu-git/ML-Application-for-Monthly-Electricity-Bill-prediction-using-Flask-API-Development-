from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

MODEL_FILE = "monthly_bill.pkl"

# Save model
def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# Load model
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

@app.route("/train", methods=["POST"])
def train():

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    X = df[["num_rooms", "num_people", "housearea", "is_ac","is_tv","is_flat","num_children","is_urban"]]
    y = df["amount_paid"]

    model = LinearRegression()
    model.fit(X, y)

    save_model(model)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return jsonify({
        "message": "Model trained successfully for Monthly Electricity Bill prediction",
        "train_mse": mse,
        "train_r2": r2
    })

@app.route("/test", methods=["POST"])
def test():

    try:
        model = load_model()
    except:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)
    X = df[["num_rooms", "num_people", "housearea", "is_ac","is_tv","is_flat","num_children","is_urban"]]
    y = df["amount_paid"]

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return jsonify({
        "test_mse": mse,
        "test_r2": r2
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
 {
        "features": [4,4,900,1,1,1,0,1]
    }
    """
    try:
        model = load_model()
    except:
        return jsonify({"error": "No trained model found. Please train first."}), 400

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    features = np.array([data["features"]])  # must be 2D
    prediction = model.predict(features)[0]

    return jsonify({"predicted_Monthly_Electricity_Bill": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
