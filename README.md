## ⚡️Household Energy Bill Predictor API (Mixed-Feature Regression)

This project implements a robust machine learning API built with Flask and Scikit-learn that predicts a household's monthly energy bill (amount_paid) based on a mix of continuous and binary features. It serves as a practical demonstration of deploying a Linear Regression model for a real-world financial forecasting task.

## 🎯 Project Goal

The primary objective is to accurately predict the continuous numerical target variable (monthly bill amount) by leveraging a dataset that combines:

Numerical Features: num_rooms, num_people, housearea, num_children.

Binary/Categorical Features: is_ac, is_tv, is_flat, is_urban.

This setup addresses a common challenge in data science: building accurate regression models with heterogeneous input data types.

## ✨ Key Features

Mixed-Feature Regression: Uses Linear Regression to handle a blend of quantitative (e.g., area) and qualitative (e.g., presence of AC) variables.

RESTful API Endpoints: Provides clear endpoints for standard ML workflow operations:

/train: Trains the model on the full dataset and returns performance metrics (R 
2
  and MSE).

/test: Evaluates the trained model against external, unseen data.

/predict: Accepts a JSON payload of household features and returns the predicted monthly bill.

Robust Data Handling: Includes essential data cleaning logic (pd.to_numeric, dropna) to ensure the model is protected from corrupted or incomplete inputs.

## ⚙️ Technology Stack

Backend Framework: Flask (Python)

Machine Learning: Scikit-learn (LinearRegression)

Data Processing: Pandas

Serialization: Pickle

## 🚀 API Endpoints & Usage

🧪 Testing with Postman

You can easily test the API endpoints using Postman (or any API client). Assume the local server is running at http://127.0.0.1:5000.

A. Train the Model (/train)
Method: POST

URL: http://127.0.0.1:5000/train

Body: Select none (the model uses the static CSV file).

Action: Send the request. This will train the model and save the monthly_bill.pkl file.

B. Get a Prediction (/predict)
Method: POST

URL: http://127.0.0.1:5000/predict

Body: Select raw and set format to JSON.

Paste: Use the example JSON payload below:

{
    "features": [4,4,900,1,1,1,0,1]
}


Action: Send the request. The response should be a JSON object containing the predicted monthly bill amount. API endpoints using Postman (or any API client).

