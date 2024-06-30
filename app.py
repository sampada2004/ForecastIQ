import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data = data.dropna()

# Prepare data for modeling
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model
model = DecisionTreeRegressor(max_depth=3, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', verbose=1)
grid_search.fit(x_train, y_train)

print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# Function to predict Units Sold based on user input
def predict_units_sold(total_price, base_price):
    features = np.array([[total_price, base_price]])
    predicted_units_sold = best_model.predict(features)[0]
    return predicted_units_sold

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    total_price = float(request.form['total-price'])
    base_price = float(request.form['base-price'])
    prediction = predict_units_sold(total_price, base_price)
    return jsonify(prediction=int(round(prediction)))

if __name__ == "__main__":
    app.run(debug=True)
